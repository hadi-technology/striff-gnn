"""Manual HGT implementation using only primitive tensor ops.

Reimplements PyG's HGTConv as pure matmul + scatter + softmax,
making it fully ONNX-exportable. The trained weights from the PyG
HGTConv transfer directly because the math is identical.

Exact flow matching PyG:
  1. kqv = W_kqv @ h → split into K, Q, V
  2. K, Q, V reshaped to (N, H, d)
  3. For each edge type, look up k_rel[edge_type_idx] and v_rel[edge_type_idx]
  4. K_rel = K[src] @ k_rel  (per-edge-type linear transform)
  5. V_rel = V[src] @ v_rel
  6. Attention: softmax_per_target(Q[tgt] * K_rel * p_rel / sqrt(d))
  7. Message = attn * V_rel, scatter_add to targets
  8. Output projection + residual with GELU activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import math

EDGE_TYPES = ["EXTENSION", "IMPLEMENTATION", "COMPOSITION", "AGGREGATION", "ASSOCIATION"]
EDGE_TYPE_MAP = {et: i for i, et in enumerate(EDGE_TYPES)}


class ManualHGTConv(nn.Module):
    """HGT convolution implemented with primitive ops for ONNX export."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 num_edge_types: int = 20):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.kqv_weight = nn.Parameter(torch.empty(3 * out_dim, in_dim))
        self.kqv_bias = nn.Parameter(torch.empty(3 * out_dim))
        self.out_weight = nn.Parameter(torch.empty(out_dim, out_dim))
        self.out_bias = nn.Parameter(torch.empty(out_dim))
        self.k_rel = nn.Parameter(torch.empty(num_edge_types, self.head_dim, self.head_dim))
        self.v_rel = nn.Parameter(torch.empty(num_edge_types, self.head_dim, self.head_dim))
        self.skip = nn.Parameter(torch.empty(1))
        # Per edge-type attention prior per head
        self.p_rel = nn.Parameter(torch.empty(len(EDGE_TYPES), num_heads))

    def forward(self, h: torch.Tensor,
                edge_list: list) -> torch.Tensor:
        N = h.size(0)
        D = self.out_dim
        H = self.num_heads
        d = self.head_dim

        # Project to K, Q, V and reshape
        kqv = F.linear(h, self.kqv_weight, self.kqv_bias)  # (N, 3*D)
        K, Q, V = kqv.split(D, dim=-1)
        K = K.view(N, H, d)
        Q = Q.view(N, H, d)
        V = V.view(N, H, d)

        # Collect all edges across types with type-aware transforms
        all_K_rel = []
        all_V_rel = []
        all_Q_tgt = []
        all_tgt_idx = []
        all_attn_prior = []

        for et_idx_in_list, (et_global_idx, ei) in enumerate(edge_list):
            src, tgt = ei[0], ei[1]
            E = src.size(0)

            # Relation-specific transforms
            k_r = self.k_rel[et_global_idx]  # (d, d)
            v_r = self.v_rel[et_global_idx]  # (d, d)

            # K_rel[src] = K[src] @ k_rel: (E, H, d) × (d, d)
            K_src = K[src]  # (E, H, d)
            K_rel = torch.einsum('ehd,df->ehf', K_src, k_r)

            # V_rel[src] = V[src] @ v_rel
            V_src = V[src]
            V_rel = torch.einsum('ehd,df->ehf', V_src, v_r)

            # Attention prior for this edge type
            prior = self.p_rel[et_global_idx]  # (H,)

            all_K_rel.append(K_rel)
            all_V_rel.append(V_rel)
            all_Q_tgt.append(Q[tgt])
            all_tgt_idx.append(tgt)
            all_attn_prior.append(prior.unsqueeze(0).expand(E, H))

        if not all_K_rel:
            # No edges — just output projection on input
            alpha = torch.sigmoid(self.skip)
            return h * (1 - alpha)

        # Concatenate all edge types
        K_rel_all = torch.cat(all_K_rel, dim=0)  # (total_E, H, d)
        V_rel_all = torch.cat(all_V_rel, dim=0)
        Q_tgt_all = torch.cat(all_Q_tgt, dim=0)
        tgt_all = torch.cat(all_tgt_idx, dim=0)  # (total_E,)
        prior_all = torch.cat(all_attn_prior, dim=0)  # (total_E, H)

        # Attention scores: Q[tgt] * K_rel / sqrt(d) * prior
        scale = float(d) ** -0.5
        alpha = (Q_tgt_all * K_rel_all).sum(dim=-1) * scale  # (total_E, H)
        alpha = alpha * prior_all  # multiply by edge-type prior

        # Softmax per target node
        # For ONNX: compute max per target, then exp, then scatter
        # Step 1: subtract max per target for numerical stability
        # Use segment-based approach
        alpha_max = torch.full((N, H), float('-inf'), device=h.device, dtype=h.dtype)
        alpha_max.scatter_reduce_(0, tgt_all.unsqueeze(1).expand_as(alpha), alpha, reduce='amax', include_self=True)
        alpha_stable = alpha - alpha_max[tgt_all]

        # Step 2: exp
        exp_alpha = torch.exp(alpha_stable)

        # Step 3: sum exp per target
        exp_sum = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        exp_sum.scatter_add_(0, tgt_all.unsqueeze(1).expand_as(exp_alpha), exp_alpha)

        # Step 4: normalize
        attn = exp_alpha / (exp_sum[tgt_all] + 1e-16)  # (total_E, H)

        # Weighted messages
        msg = attn.unsqueeze(-1) * V_rel_all  # (total_E, H, d)

        # Scatter-add to targets
        out = torch.zeros(N, H, d, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, tgt_all.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        # Reshape: (N, H, d) → (N, D)
        out = out.reshape(N, D)

        # GELU activation (matching PyG HGTConv)
        out = F.gelu(out)

        # Output projection
        out = F.linear(out, self.out_weight, self.out_bias)

        # Residual
        alpha_skip = torch.sigmoid(self.skip)
        out = alpha_skip * out + (1 - alpha_skip) * h

        return out


class ExportableEncoder(nn.Module):
    """Encoder with manual HGTConv that exports to ONNX."""

    def __init__(self, input_dim: int = 404, hidden_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            ManualHGTConv(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, node_features: torch.Tensor,
                ext_ei: torch.Tensor, impl_ei: torch.Tensor,
                comp_ei: torch.Tensor, agg_ei: torch.Tensor,
                assoc_ei: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(node_features))

        # Always prepend a self-loop (0→0) to each edge type.
        # This ensures every edge type has >=1 edge, which prevents
        # TorchScript tracing from baking in empty-tensor branches
        # and causing ONNX Runtime shape mismatches at inference.
        self_loop = torch.zeros(2, 1, dtype=torch.long, device=node_features.device)

        edge_list = [
            (EDGE_TYPE_MAP["EXTENSION"], torch.cat([self_loop, ext_ei], dim=1)),
            (EDGE_TYPE_MAP["IMPLEMENTATION"], torch.cat([self_loop, impl_ei], dim=1)),
            (EDGE_TYPE_MAP["COMPOSITION"], torch.cat([self_loop, comp_ei], dim=1)),
            (EDGE_TYPE_MAP["AGGREGATION"], torch.cat([self_loop, agg_ei], dim=1)),
            (EDGE_TYPE_MAP["ASSOCIATION"], torch.cat([self_loop, assoc_ei], dim=1)),
        ]

        for conv in self.convs:
            h = conv(h, edge_list)
            h = torch.relu(h)

        return h


def transfer_weights(src_encoder, dst_encoder):
    """Transfer trained weights from PyG HGTConv to ManualHGTConv."""
    dst_encoder.input_proj.weight.data.copy_(src_encoder.input_proj.weight.data)
    dst_encoder.input_proj.bias.data.copy_(src_encoder.input_proj.bias.data)

    for i, (src_conv, dst_conv) in enumerate(zip(src_encoder.convs, dst_encoder.convs)):
        dst_conv.kqv_weight.data.copy_(src_conv.kqv_lin.lins.node.weight.data)
        dst_conv.kqv_bias.data.copy_(src_conv.kqv_lin.lins.node.bias.data)
        dst_conv.out_weight.data.copy_(src_conv.out_lin.lins.node.weight.data)
        dst_conv.out_bias.data.copy_(src_conv.out_lin.lins.node.bias.data)
        dst_conv.k_rel.data.copy_(src_conv.k_rel.weight.data)
        dst_conv.v_rel.data.copy_(src_conv.v_rel.weight.data)
        dst_conv.skip.data.copy_(src_conv.skip.node.data)
        for et in EDGE_TYPES:
            key = f"node__{et}__node"
            if hasattr(src_conv.p_rel, key):
                dst_conv.p_rel.data[EDGE_TYPE_MAP[et]].copy_(
                    src_conv.p_rel[key].data.squeeze(0))

        print(f"  Layer {i}: weights transferred")


def export_encoder_onnx(model_path, output_dir, hidden_dim=128, num_layers=3, num_heads=4):
    """Export the trained encoder to ONNX via manual HGT implementation."""
    from src.model.graph_mae import ArchGraphMAE

    output_path = Path(output_dir)

    teacher = ArchGraphMAE(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state)
    teacher.eval()

    exportable = ExportableEncoder(input_dim=404, hidden_dim=hidden_dim,
                                   num_layers=num_layers, num_heads=num_heads)
    print("Transferring weights...")
    transfer_weights(teacher.encoder, exportable)
    exportable.eval()

    # Verify parity
    print("\nVerifying output parity...")
    N = 50
    sample_x = torch.randn(N, 404)
    sample_edges = [torch.randint(0, N, (2, 20)) for _ in range(5)]

    with torch.no_grad():
        x_dict = {"node": sample_x}
        edge_index_dict = {("node", et, "node"): sample_edges[i] for i, et in enumerate(EDGE_TYPES)}
        teacher_out = teacher.encoder(x_dict, edge_index_dict)
        export_out = exportable(sample_x, *sample_edges)

    diff = (teacher_out - export_out).abs()
    cos = F.cosine_similarity(teacher_out, export_out, dim=-1).mean().item()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Cosine sim: {cos:.4f}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        exportable,
        (sample_x, *sample_edges),
        str(output_path / "arch_scorer.onnx"),
        opset_version=17,
        input_names=["node_features"] + [f"edge_index_{et.lower()}" for et in EDGE_TYPES],
        output_names=["node_embeddings"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            **{f"edge_index_{et.lower()}": {1: "num_edges"} for et in EDGE_TYPES},
        },
        export_params=True,
        dynamo=False,
    )

    import os
    size = os.path.getsize(str(output_path / "arch_scorer.onnx"))
    print(f"ONNX export: {size:,} bytes ({size/1024/1024:.1f} MB)")

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path / "arch_scorer.onnx"))

    for n in [10, 50, 200, 500]:
        f = np.random.randn(n, 404).astype(np.float32)
        edges = [np.random.randint(0, n, (2, max(1, n // 5))).astype(np.int64) for _ in range(5)]
        feed = {"node_features": f}
        for i, et in enumerate(EDGE_TYPES):
            feed[f"edge_index_{et.lower()}"] = edges[i]
        out = sess.run(None, feed)[0]
        print(f"  Size {n}: output shape {out.shape}")

    # Test with empty edge types (critical for ONNX tracing fix)
    print("\nTesting with empty edge types...")
    n = 30
    f = np.random.randn(n, 404).astype(np.float32)
    feed = {"node_features": f}
    for i, et in enumerate(EDGE_TYPES):
        if i < 2:
            feed[f"edge_index_{et.lower()}"] = np.random.randint(0, n, (2, 10)).astype(np.int64)
        else:
            feed[f"edge_index_{et.lower()}"] = np.zeros((2, 0), dtype=np.int64)
    out = sess.run(None, feed)[0]
    print(f"  Mixed edges: output shape {out.shape} ✓")

    # Test with ALL empty edge types
    feed2 = {"node_features": np.random.randn(5, 404).astype(np.float32)}
    for et in EDGE_TYPES:
        feed2[f"edge_index_{et.lower()}"] = np.zeros((2, 0), dtype=np.int64)
    out2 = sess.run(None, feed2)[0]
    print(f"  All empty: output shape {out2.shape} ✓")

    # PyTorch vs ONNX diff
    with torch.no_grad():
        pt_out = exportable(sample_x, *sample_edges).numpy()
    onnx_out = sess.run(None, {
        "node_features": sample_x.numpy(),
        **{f"edge_index_{et.lower()}": sample_edges[i].numpy() for i, et in enumerate(EDGE_TYPES)}
    })[0]
    print(f"\nPyTorch vs ONNX max diff: {np.abs(pt_out - onnx_out).max():.8f}")
    print("Done!")


if __name__ == "__main__":
    export_encoder_onnx(
        model_path="data/models/best_model.pt",
        output_dir="data/models",
    )
