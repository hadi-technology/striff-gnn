#!/usr/bin/env python3
"""Re-normalize existing graph JSON files to add synthetic module nodes
and update to 404-dim features.

This script reads the existing normalized graph JSON files, adds synthetic
MODULE nodes for Python/TS source files, rebuilds the PyG datasets with
404-dim features, and saves both updated files.

Does NOT require clarpse Docker or re-parsing source code.
"""

import sys
import os
import gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from src.corpus.normalizer import _add_synthetic_modules
from src.graph.dataset import graph_to_hetero_data, save_dataset


def renormalize_graph(graph: dict, language: str) -> dict:
    """Add synthetic module nodes and synthetic flags to existing graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    metrics = graph.get("metrics", {})

    # Add synthetic field to existing nodes
    for node in nodes:
        if "synthetic" not in node:
            node["synthetic"] = False

    # Add synthetic modules for Python/TS
    if language.lower() in ("python", "typescript"):
        _add_synthetic_modules(nodes, edges, metrics)

    return {"nodes": nodes, "edges": edges, "metrics": metrics}


def main():
    data_dir = Path(os.environ.get("STRIFF_DATA_DIR", "./data"))
    graph_dir = data_dir / "corpus" / "graphs"
    pt_dir = data_dir / "corpus" / "pt"

    if not graph_dir.exists():
        print(f"ERROR: Graph directory not found: {graph_dir}")
        sys.exit(1)

    graph_files = sorted(graph_dir.glob("*.json"))
    print(f"Found {len(graph_files)} graph files to re-normalize")

    updated = 0
    synthetic_count = 0
    failed = 0

    for gf in graph_files:
        # Determine language from filename
        name = gf.stem
        if name.startswith("python_"):
            lang = "python"
        elif name.startswith("typescript_"):
            lang = "typescript"
        else:
            lang = "java"

        print(f"  {name} ({lang})...", end=" ", flush=True)

        try:
            with open(gf) as f:
                graph = json.load(f)

            old_nodes = len(graph.get("nodes", []))
            graph = renormalize_graph(graph, lang)
            new_nodes = len(graph.get("nodes", []))

            # Count synthetic nodes
            synth = sum(1 for n in graph["nodes"] if n.get("synthetic", False))
            synthetic_count += synth

            # Save updated graph
            with open(gf, "w") as f:
                json.dump(graph, f)

            # Rebuild PyG dataset with 404-dim features
            pt_file = pt_dir / f"{name}.pt"
            data = graph_to_hetero_data(graph, language=lang, repo_name=name)
            del graph
            save_dataset(data, pt_file)
            del data
            gc.collect()

            diff = new_nodes - old_nodes
            print(f"{old_nodes}→{new_nodes} nodes (+{diff} synthetic) ✓")
            updated += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDone! Updated: {updated}, Failed: {failed}, Total synthetic: {synthetic_count}")


if __name__ == "__main__":
    main()
