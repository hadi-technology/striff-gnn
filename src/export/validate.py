"""Validate ONNX model outputs match PyTorch model within tolerance."""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

from .to_onnx import ExportableEncoder
from ..model.graph_mae import ArchGraphMAE, EDGE_TYPES


def validate_onnx(
    model_path: str = "artifacts/models/best_model.pt",
    onnx_path: str = "artifacts/models/arch_scorer.onnx",
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    max_relative_error: float = 1e-4,
    num_samples: int = 100,
):
    # Load PyTorch model
    model = ArchGraphMAE(hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    exportable = ExportableEncoder(model)

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    max_error = 0.0
    failures = 0

    for i in range(num_samples):
        num_nodes = np.random.randint(10, 100)
        node_features = torch.randn(num_nodes, 403)

        edge_indices = []
        feed = {"node_features": node_features.numpy()}
        for j, et in enumerate(EDGE_TYPES):
            num_edges = np.random.randint(0, 50)
            if num_edges > 0:
                idx = torch.randint(0, num_nodes, (2, num_edges))
            else:
                idx = torch.zeros((2, 0), dtype=torch.long)
            edge_indices.append(idx)
            feed[f"edge_index_{et.lower()}"] = idx.numpy()

        # PyTorch inference
        with torch.no_grad():
            pt_output = exportable(node_features, *edge_indices).numpy()

        # ONNX inference
        onnx_output = session.run(None, feed)[0]

        # Compare
        rel_error = np.max(np.abs(pt_output - onnx_output) / (np.abs(pt_output) + 1e-8))
        max_error = max(max_error, rel_error)

        if rel_error > max_relative_error:
            failures += 1
            print(f"Sample {i}: relative error {rel_error:.6f} exceeds threshold")

    print(f"\nValidation complete: max_relative_error={max_error:.6f}, failures={failures}/{num_samples}")
    if failures > 0:
        print("FAILED: ONNX outputs diverge from PyTorch")
        return False
    print("PASSED: ONNX outputs match PyTorch within tolerance")
    return True


if __name__ == "__main__":
    validate_onnx()
