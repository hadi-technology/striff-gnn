#!/usr/bin/env python3
"""Build corpus: parse cloned repos using clarpse Docker, normalize, build PyG datasets.

All data stored under STRIFF_DATA_DIR (default: ./data).
Resumable: skips repos that already have parsed graph files.
Processes repos one at a time to avoid memory spikes.

Requires clarpse Docker container running on CLARPSE_API_URL (default: http://localhost:9080).
"""

import sys
import os
import gc
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from src.corpus.cloner import clone_all, STRIFF_DATA_DIR
from src.corpus.parser_client import parse_repo, health_check
from src.corpus.normalizer import normalize, save_normalized
from src.graph.dataset import graph_to_hetero_data, save_dataset

# No repos skipped — process everything
SKIP_REPOS = set()


def main():
    data_dir = Path(os.environ.get("STRIFF_DATA_DIR", STRIFF_DATA_DIR))

    # Check clarpse server is running
    if not health_check():
        print("ERROR: Clarpse server is not running!")
        print("  Start it with: docker run -p 9080:8080 clarpse-server")
        print(f"  Expected at: {os.environ.get('CLARPSE_API_URL', 'http://localhost:9080')}")
        sys.exit(1)
    print("Clarpse server is healthy")

    # Step 1: Ensure repos are cloned (resumable)
    print("=" * 60)
    print("STEP 1: Ensuring repos are cloned")
    print("=" * 60)
    repos = clone_all(str(data_dir / "corpus" / "repos"))
    print(f"Available repos: {len(repos)}")

    # Step 2: Parse repos one at a time
    print("\n" + "=" * 60)
    print("STEP 2: Parsing repos with clarpse (one at a time)")
    print("=" * 60)
    graph_dir = data_dir / "corpus" / "graphs"
    pt_dir = data_dir / "corpus" / "pt"
    graph_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)

    parsed = 0
    skipped = 0
    failed = 0
    skipped_large = 0

    for key, repo_path in repos.items():
        if key in SKIP_REPOS:
            skipped_large += 1
            continue

        lang = key.split("/")[0]
        safe_name = key.replace("/", "_")
        graph_file = graph_dir / f"{safe_name}.json"
        pt_file = pt_dir / f"{safe_name}.pt"

        # Resume: skip if already parsed
        if graph_file.exists() and pt_file.exists():
            skipped += 1
            continue

        print(f"\n  [{time.strftime('%H:%M:%S')}] Parsing {key} ({lang})...",
              end=" ", flush=True)
        start = time.time()

        result = parse_repo(repo_path, lang)

        if result is None:
            print(f"FAILED ({time.time()-start:.0f}s)")
            failed += 1
            gc.collect()
            continue

        # Normalize and save
        graph = normalize(result, language=lang)
        num_nodes = len(graph["nodes"])
        num_edges = len(graph["edges"])
        elapsed = time.time() - start
        print(f"{num_nodes} nodes, {num_edges} edges ({elapsed:.0f}s)")

        # Free the raw result immediately
        del result
        gc.collect()

        if num_nodes < 3:
            print(f"  Skipping {key}: too few nodes ({num_nodes})")
            continue

        save_normalized(graph, graph_file)

        # Build PyG data and free graph immediately
        data = graph_to_hetero_data(graph, language=lang, repo_name=safe_name)
        del graph
        save_dataset(data, pt_file)
        del data
        gc.collect()
        parsed += 1

    print(f"\nDone! Parsed: {parsed}, Skipped (existing): {skipped}, "
          f"Skipped (too large): {skipped_large}, Failed: {failed}")
    print(f"Graph files: {len(list(graph_dir.glob('*.json')))}")
    print(f"PyG files: {len(list(pt_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
