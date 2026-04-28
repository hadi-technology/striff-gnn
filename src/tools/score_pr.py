"""Local end-to-end test: score a PR using a trained model."""

import json
import sys
import torch
import numpy as np
from pathlib import Path

from ..model.graph_mae import ArchGraphMAE, EDGE_TYPES
from ..graph.dataset import graph_to_hetero_data, load_dataset
from ..graph.features import NODE_FEATURE_DIM


def score_pr(
    graph_path: str,
    model_path: str = "artifacts/models/best_model.pt",
    seed_file: str = None,
):
    graph = load_dataset(Path(graph_path))
    if not graph:
        print(f"No dataset found at {graph_path}")
        return

    model = ArchGraphMAE()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    x_dict = {"node": graph["node"].x}
    edge_index_dict = {}
    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key in graph.edge_types:
            edge_index_dict[key] = graph[key].edge_index

    with torch.no_grad():
        h = model.encode(x_dict, edge_index_dict)

    # Anomaly score: L2 norm of embedding per node
    norms = torch.norm(h, dim=1).numpy()

    print(f"Node embeddings shape: {h.shape}")
    print(f"Anomaly scores (L2 norm):")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std:  {norms.std():.4f}")
    print(f"  Min:  {norms.min():.4f}")
    print(f"  Max:  {norms.max():.4f}")

    # Top anomalous nodes
    if hasattr(graph["node"], "node_ids"):
        node_ids = graph["node"].node_ids
        top_indices = np.argsort(norms)[-10:][::-1]
        print(f"\nTop 10 anomalous components:")
        for idx in top_indices:
            if idx < len(node_ids):
                print(f"  {node_ids[idx]}: {norms[idx]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.score_pr <dataset.pt>")
        sys.exit(1)
    score_pr(sys.argv[1])
