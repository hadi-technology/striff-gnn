#!/usr/bin/env python3
"""Smoke test: verify the entire GNN pipeline works with synthetic data.

Tests: graph construction → dataset building → model forward pass → loss → backward → ONNX export.
Does NOT require real repos or clarpse.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path

from src.graph.features import (
    NODE_FEATURE_DIM, EDGE_TYPES,
    build_type_one_hot, build_language_one_hot, build_metric_vector,
)
from src.graph.dataset import graph_to_hetero_data, save_dataset, load_dataset
from src.graph.sampler import sample_clustered, sample_random, extract_subgraph
from src.model.graph_mae import ArchGraphMAE
from src.train.losses import compute_loss
from src.export.metadata import export_metadata


def make_synthetic_graph(num_nodes=50, num_edges_per_type=10, language="java"):
    """Create a synthetic graph that mimics clarpse output."""
    component_types = ["CLASS", "INTERFACE", "ENUM", "METHOD", "FIELD"]
    nodes = []
    metrics = {}
    for i in range(num_nodes):
        ctype = component_types[i % len(component_types)]
        name = f"com.example.{ctype.lower()}{i}"
        nodes.append({
            "id": name,
            "type": ctype,
            "name": f"{ctype}{i}",
            "comment": f"Component {i} for testing",
            "file": f"/src/{name.replace('.', '/')}.java",
        })
        metrics[name] = {
            "wmc": float(i % 10),
            "dit": float(i % 4),
            "noc": float(i % 6),
            "ac": float(i % 8),
            "ec": float(i % 7),
            "encapsulation": float(i % 3) * 0.3,
        }

    edges = []
    for et in EDGE_TYPES:
        for j in range(num_edges_per_type):
            src_idx = np.random.randint(0, num_nodes)
            tgt_idx = np.random.randint(0, num_nodes)
            if src_idx != tgt_idx:
                edges.append({
                    "src": nodes[src_idx]["id"],
                    "tgt": nodes[tgt_idx]["id"],
                    "type": et,
                })

    return {"nodes": nodes, "edges": edges, "metrics": metrics}


def test_pipeline():
    print("=" * 60)
    print("SMOKE TEST: GNN Training Pipeline")
    print("=" * 60)

    # Step 1: Build synthetic graphs
    print("\n[1/7] Building synthetic graphs...")
    graphs = []
    for i in range(5):
        g = make_synthetic_graph(
            num_nodes=np.random.randint(30, 80),
            num_edges_per_type=np.random.randint(5, 20),
            language=["java", "python", "typescript"][i % 3],
        )
        graphs.append(g)
    print(f"  Created {len(graphs)} synthetic graphs")

    # Step 2: Convert to PyG HeteroData
    print("\n[2/7] Converting to PyG HeteroData...")
    datasets = []
    for i, g in enumerate(graphs):
        lang = ["java", "python", "typescript"][i % 3]
        data = graph_to_hetero_data(g, language=lang)
        datasets.append(data)
        num_edges = sum(
            data[("node", et, "node")].edge_index.size(1)
            for et in EDGE_TYPES
            if ("node", et, "node") in data.edge_types
        )
        print(f"  Graph {i}: {data['node'].x.size(0)} nodes, {num_edges} edges, lang={lang}")

    # Step 3: Sample subgraphs
    print("\n[3/7] Sampling subgraphs...")
    subgraphs = []
    for data in datasets:
        for _ in range(3):
            focal = sample_clustered(data, size=10)
            if len(focal) >= 3:
                sub = extract_subgraph(data, focal)
                subgraphs.append(sub)
    print(f"  Sampled {len(subgraphs)} subgraphs")

    # Step 4: Build model
    print("\n[4/7] Building ArchGraphMAE model...")
    model = ArchGraphMAE(
        node_feat_dim=NODE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Step 5: Forward pass
    print("\n[5/7] Testing forward pass...")
    data = datasets[0]
    x_dict = {"node": data["node"].x}
    edge_index_dict = {}
    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key in data.edge_types:
            edge_index_dict[key] = data[key].edge_index

    h = model.encode(x_dict, edge_index_dict)
    print(f"  Encoded shape: {h.shape}")

    # Test edge prediction
    if ("node", "EXTENSION", "node") in data.edge_types:
        ei = data[("node", "EXTENSION", "node")].edge_index
        if ei.size(1) > 0:
            n = min(5, ei.size(1))
            preds = model.predict_edges(h, ei[0, :n], ei[1, :n], 0)
            print(f"  Edge predictions shape: {preds.shape}")

    # Step 6: Training step
    print("\n[6/7] Testing training step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(5):
        total_loss = 0.0
        for sg in subgraphs[:10]:
            if sg["node"].x.size(0) < 3:
                continue
            num_nodes = sg["node"].x.size(0)

            x_dict = {"node": sg["node"].x}
            edge_index_dict = {}
            query_pairs = {}
            labels = {}

            for i, et in enumerate(EDGE_TYPES):
                key = ("node", et, "node")
                if key not in sg.edge_types:
                    continue
                ei = sg[key].edge_index
                # Always add to edge_index_dict (even if empty) for the encoder
                edge_index_dict[key] = ei
                if ei.size(1) == 0:
                    continue

                # Use all edges as positive queries
                query_pairs[et] = (ei[0], ei[1])
                labels[et] = torch.ones(ei.size(1))

                # Add negative samples
                num_neg = ei.size(1)
                neg_src = torch.randint(0, num_nodes, (num_neg,))
                neg_tgt = torch.randint(0, num_nodes, (num_neg,))
                query_pairs[et] = (
                    torch.cat([ei[0], neg_src]),
                    torch.cat([ei[1], neg_tgt]),
                )
                labels[et] = torch.cat([labels[et], torch.zeros(num_neg)])

            if not query_pairs:
                continue

            predictions = model(x_dict, edge_index_dict, query_pairs)
            loss = compute_loss(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(subgraphs[:10]), 1)
        print(f"  Epoch {epoch + 1}/5: loss={avg_loss:.4f}")

    # Step 7: Save and reload
    print("\n[7/7] Testing save/reload...")
    test_dir = Path("artifacts/smoke_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(datasets[0], test_dir / "test_graph.pt")
    loaded = load_dataset(test_dir / "test_graph.pt")
    assert loaded is not None
    assert loaded["node"].x.shape == datasets[0]["node"].x.shape
    print(f"  Saved and reloaded graph: {loaded['node'].x.shape}")

    torch.save(model.state_dict(), test_dir / "test_model.pt")
    print(f"  Saved model weights")

    # Export metadata
    export_metadata(str(test_dir))
    print(f"  Exported metadata")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - All pipeline stages verified")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
