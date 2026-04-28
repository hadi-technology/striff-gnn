#!/usr/bin/env python3
"""Build dataset: sample subgraphs from parsed repos.

Processes one repo at a time to limit memory usage.
Resumable: skips repos that already have subgraph files.
"""

import sys
import os
import gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import yaml
from pathlib import Path
from src.graph.dataset import load_dataset, save_dataset
from src.graph.sampler import sample_clustered, sample_random, extract_subgraph


def main():
    base = Path(os.environ.get("STRIFF_DATA_DIR", "./data"))
    data_dir = base / "corpus" / "pt"
    output_dir = base / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("config/training.yaml") as f:
        config = yaml.safe_load(f)

    ds_config = config["dataset"]
    clustered = ds_config["clustered_samples"]
    random_samples = ds_config["random_samples"]
    min_size = ds_config["min_focal_size"]
    max_size = ds_config["max_focal_size"]

    data_files = sorted(data_dir.glob("*.pt"))
    print(f"Found {len(data_files)} dataset files")

    # Find already-processed repo names for resume support
    existing = set()
    for f in output_dir.glob("*.pt"):
        name = f.stem.rsplit("_clustered_", 1)[0] if "_clustered_" in f.stem else f.stem.rsplit("_random_", 1)[0]
        existing.add(name)
    print(f"Already processed: {len(existing)} repos")

    total_subgraphs = len(list(output_dir.glob("*.pt")))
    processed = 0

    for data_file in data_files:
        name = data_file.stem
        if name in existing:
            print(f"  Skipping {name} (already sampled)")
            continue

        # Load one repo at a time
        data = load_dataset(data_file)
        if data is None or data["node"].x.size(0) < min_size:
            print(f"  Skipping {name} (too few nodes)")
            del data
            continue

        num_nodes = data["node"].x.size(0)
        print(f"  Sampling from {name} ({num_nodes} nodes)...", end=" ", flush=True)
        count = 0

        for i in range(clustered):
            size = random.randint(min_size, min(max_size, num_nodes))
            focal = sample_clustered(data, size)
            if len(focal) < min_size:
                continue
            sub = extract_subgraph(data, focal)
            save_dataset(sub, output_dir / f"{name}_clustered_{i}.pt")
            del sub
            count += 1

        for i in range(random_samples):
            size = random.randint(min_size, min(max_size, num_nodes))
            focal = sample_random(data, size)
            if len(focal) < min_size:
                continue
            sub = extract_subgraph(data, focal)
            save_dataset(sub, output_dir / f"{name}_random_{i}.pt")
            del sub
            count += 1

        total_subgraphs += count
        processed += 1
        print(f"{count} subgraphs")

        # Free memory before next repo
        del data
        gc.collect()

    print(f"\nTotal subgraphs: {total_subgraphs} (newly processed: {processed})")


if __name__ == "__main__":
    main()
