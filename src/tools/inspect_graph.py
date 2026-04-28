"""Debug tool: inspect a parsed repo graph."""

import json
import sys
from pathlib import Path

from ..corpus.normalizer import load_normalized
from ..graph.dataset import graph_to_hetero_data
from ..graph.features import EDGE_TYPES


def inspect(graph_path: str):
    graph = load_normalized(Path(graph_path))
    if not graph:
        print(f"No graph found at {graph_path}")
        return

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    metrics = graph.get("metrics", {})

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Nodes with metrics: {len(metrics)}")

    # Edge type distribution
    type_counts = {}
    for e in edges:
        t = e["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nEdge types:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Component type distribution
    comp_types = {}
    for n in nodes:
        t = n["type"]
        comp_types[t] = comp_types.get(t, 0) + 1
    print(f"\nComponent types:")
    for t, c in sorted(comp_types.items()):
        print(f"  {t}: {c}")

    # Convert to PyG and inspect
    data = graph_to_hetero_data(graph, "java")
    print(f"\nPyG HeteroData:")
    print(f"  Node features shape: {data['node'].x.shape}")
    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key in data.edge_types:
            print(f"  {et} edges: {data[key].edge_index.size(1)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.inspect_graph <graph.json>")
        sys.exit(1)
    inspect(sys.argv[1])
