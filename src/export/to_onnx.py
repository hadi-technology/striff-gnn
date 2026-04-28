"""ONNX-exportable graph scorer with real message passing.

Unlike the feature-only MLP, this model takes both node features AND
an adjacency matrix, performing GCN-style message passing via matmul.
This preserves the graph structure signal at inference time.

Architecture:
  input: node_features (N, 403), adj_matrix (N, N)
  for each layer:
    h_msg = adj @ h @ W      (message passing = matmul, ONNX-compatible)
    h = norm(h + h_msg)       (residual + layernorm)
    h = h + ff(h)             (feed-forward)
  output: sigmoid(score_head(h))  (per-node anomaly score)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from ..model.graph_mae import ArchGraphMAE, EDGE_TYPES
from ..graph.features import NODE_FEATURE_DIM


class ONNXGraphScorer(nn.Module):
    """Graph scorer that uses adjacency matrix for real message passing.

    Takes (node_features, adj_matrix) as input. The adjacency matrix
    enables GCN-style neighborhood aggregation — a true graph model.
    """

    def __init__(self, input_dim: int = 403, hidden_dim: int = 128,
                 num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "W_msg": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "ff1": nn.Linear(hidden_dim, hidden_dim * 2),
                "ff2": nn.Linear(hidden_dim * 2, hidden_dim),
                "norm": nn.LayerNorm(hidden_dim),
            }))
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """Score nodes using graph structure.

        Args:
            node_features: (N, 403) node feature matrix
            adj_matrix: (N, N) normalized adjacency matrix (with self-loops)

        Returns:
            scores: (N,) anomaly score per node in [0, 1]
        """
        h = torch.relu(self.input_proj(node_features))
        for layer in self.layers:
            # Message passing: aggregate neighbors via adjacency
            h_msg = torch.matmul(adj_matrix, h)  # (N, N) @ (N, D) = (N, D)
            h_msg = layer["W_msg"](h_msg)
            h = layer["norm"](h + h_msg)
            # Feed-forward
            h2 = torch.relu(layer["ff1"](h))
            h2 = layer["ff2"](h2)
            h = h + h2
        return self.score_head(h).squeeze(-1)


def build_adj_matrix(edge_indices: dict[str, torch.Tensor],
                     num_nodes: int) -> torch.Tensor:
    """Build a symmetric normalized adjacency matrix with self-loops.

    Args:
        edge_indices: dict of edge_type -> (2, E) tensor
        num_nodes: number of nodes

    Returns:
        adj: (N, N) normalized adjacency matrix
    """
    adj = torch.eye(num_nodes, dtype=torch.float32)
    for et, ei in edge_indices.items():
        src, tgt = ei[0], ei[1]
        for s, t in zip(src.tolist(), tgt.tolist()):
            adj[s, t] = 1.0
            adj[t, s] = 1.0  # undirected

    # D^{-1/2} A D^{-1/2} normalization
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
    adj_norm = adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)
    return adj_norm


def distill_graph_scorer(
    teacher_model: ArchGraphMAE,
    scorer: ONNXGraphScorer,
    datasets: list,
    device: str = "cpu",
    epochs: int = 15,
    lr: float = 1e-3,
) -> ONNXGraphScorer:
    """Distill the GNN encoder into the graph scorer.

    Trains the scorer to reproduce the teacher's per-node anomaly patterns,
    using both node features and adjacency structure.
    """
    teacher_model.eval()
    scorer.train()
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for data in datasets:
            num_nodes = data["node"].x.size(0)
            if num_nodes < 5:
                continue

            # Build adjacency
            edge_indices = {}
            for et in EDGE_TYPES:
                key = ("node", et, "node")
                if key in data.edge_types:
                    edge_indices[et] = data[key].edge_index
            adj = build_adj_matrix(edge_indices, num_nodes)

            # Get teacher targets
            with torch.no_grad():
                x_dict = {"node": data["node"].x}
                edge_index_dict = {}
                for et in EDGE_TYPES:
                    key = ("node", et, "node")
                    if key in data.edge_types:
                        edge_index_dict[key] = data[key].edge_index

                teacher_emb = teacher_model.encode(x_dict, edge_index_dict)
                # Target: per-node anomaly score from embedding distribution
                norms = torch.norm(teacher_emb, dim=-1)
                mean_norm = norms.mean()
                targets = torch.abs(norms - mean_norm) / (norms.std() + 1e-8)
                targets = torch.clamp(targets / (targets.max() + 1e-8), 0, 1)

            # Student forward
            scores = scorer(data["node"].x, adj)
            loss = loss_fn(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg = total_loss / max(count, 1)
        print(f"  Distill epoch {epoch+1}/{epochs}: loss={avg:.4f}")

    return scorer


def export_to_onnx(
    model_path: str = "artifacts/models/best_model.pt",
    output_dir: str = "artifacts/models",
    hidden_dim: int = 128,
    num_layers: int = 3,
    distill_data_dir: str | None = None,
):
    """Export a trained model as an ONNX graph scorer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scorer = ONNXGraphScorer(
        input_dim=NODE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if distill_data_dir:
        from ..graph.dataset import load_dataset
        teacher = ArchGraphMAE(hidden_dim=hidden_dim, num_layers=num_layers,
                               num_heads=4)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        teacher.load_state_dict(state)
        teacher.eval()

        data_files = sorted(Path(distill_data_dir).glob("*.pt"))
        datasets = [load_dataset(f) for f in data_files
                     if load_dataset(f) is not None]
        print(f"Distilling from teacher into graph scorer ({len(datasets)} datasets)...")
        scorer = distill_graph_scorer(teacher, scorer, datasets, epochs=15)

    # Export to ONNX (legacy TorchScript for proper dynamic axes)
    scorer.eval()
    num_nodes = 50
    sample_features = torch.randn(num_nodes, NODE_FEATURE_DIM)
    sample_adj = torch.eye(num_nodes) + torch.rand(num_nodes, num_nodes) * 0.1
    sample_adj = (sample_adj + sample_adj.T) / 2

    torch.onnx.export(
        scorer,
        (sample_features, sample_adj),
        str(output_path / "arch_scorer.onnx"),
        opset_version=17,
        input_names=["node_features", "adj_matrix"],
        output_names=["anomaly_scores"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "adj_matrix": {0: "num_nodes", 1: "num_nodes"},
            "anomaly_scores": {0: "num_nodes"},
        },
        export_params=True,
        dynamo=False,
    )
    print(f"Exported ONNX graph scorer to {output_path / 'arch_scorer.onnx'}")

    # Validate
    validate_onnx(output_path / "arch_scorer.onnx", scorer,
                  sample_features, sample_adj)


def validate_onnx(onnx_path: Path, torch_model: nn.Module,
                  features: torch.Tensor, adj: torch.Tensor):
    """Compare ONNX and PyTorch outputs."""
    import onnxruntime as ort
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(features, adj).numpy()

    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {
        "node_features": features.numpy(),
        "adj_matrix": adj.numpy(),
    })[0]

    diff = np.abs(torch_out - onnx_out).max()
    print(f"ONNX validation: max_diff={diff:.6f} "
          f"({'PASS' if diff < 1e-4 else 'WARN: diff > 1e-4'})")

    # Test different sizes
    for n in [10, 100, 500]:
        f = np.random.randn(n, NODE_FEATURE_DIM).astype(np.float32)
        a = np.eye(n, dtype=np.float32)
        out = sess.run(None, {"node_features": f, "adj_matrix": a})[0]
        print(f"  Size {n}: output shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
