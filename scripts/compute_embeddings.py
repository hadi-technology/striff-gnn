#!/usr/bin/env python3
"""Compute sentence-transformers embeddings for all nodes in the corpus.

Loads each graph JSON file, builds text representations for every node
(name + comment), encodes them in batches using all-MiniLM-L6-v2, and
saves per-repo embedding files.

Output layout:
    {corpus_dir}/embeddings/{repo_name}.npz
Each file contains:
    - 'ids':      array of node id strings
    - 'embeddings': float32 array of shape (N, 384)

Resumable: skips repos that already have an embeddings file.
"""

import sys
import os
import gc
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path


def build_text_for_node(node: dict) -> str:
    """Build the text to encode for a single node.

    Uses the short name plus the first 200 chars of comment (if present).
    For synthetic module nodes, includes the comment as-is since it already
    describes the module.
    """
    name = node.get("name", "")
    comment = node.get("comment", "")

    if comment:
        return f"{name} {comment}"
    return name


def load_graph(graph_path: Path) -> list[dict]:
    """Load a normalized graph JSON and return its node list."""
    with open(graph_path) as f:
        graph = json.load(f)
    return graph.get("nodes", [])


def compute_repo_embeddings(nodes: list[dict], model, batch_size: int = 256) -> tuple[list[str], np.ndarray]:
    """Compute embeddings for all nodes in a single repo.

    Args:
        nodes: list of node dicts from the normalized graph
        model: sentence-transformers model
        batch_size: number of texts to encode per batch

    Returns:
        (ids, embeddings) where embeddings is float32 (N, 384)
    """
    ids = []
    texts = []
    for node in nodes:
        ids.append(node["id"])
        texts.append(build_text_for_node(node))

    # Batch encode using sentence-transformers (GPU-accelerated if available)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize for cosine similarity downstream
        convert_to_numpy=True,
    )

    return ids, embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute text embeddings for graph nodes")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=os.environ.get("STRIFF_DATA_DIR", "./data"),
        help="Root data directory (default: STRIFF_DATA_DIR env var or ./data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for sentence-transformers (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if they already exist",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Only process specific repo names (e.g. java_guava python_django). "
             "Default: process all repos.",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    graph_dir = corpus_dir / "corpus" / "graphs"
    emb_dir = corpus_dir / "corpus" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    if not graph_dir.exists():
        print(f"ERROR: Graph directory not found: {graph_dir}")
        print("Run build_corpus.py first.")
        sys.exit(1)

    # Collect graph files
    graph_files = sorted(graph_dir.glob("*.json"))
    if not graph_files:
        print("ERROR: No graph JSON files found.")
        sys.exit(1)

    # Filter to specific repos if requested
    if args.repos:
        requested = set(args.repos)
        graph_files = [gf for gf in graph_files if gf.stem in requested]
        if not graph_files:
            print(f"ERROR: None of the requested repos found: {requested}")
            print(f"Available: {[gf.stem for gf in sorted(graph_dir.glob('*.json'))]}")
            sys.exit(1)

    print(f"Found {len(graph_files)} graph files")
    print(f"Embeddings will be saved to: {emb_dir}")

    # Load sentence-transformers model once
    print("\nLoading sentence-transformers model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded on {device}")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Statistics
    total_nodes = 0
    total_with_comment = 0
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for i, gf in enumerate(graph_files):
        repo_name = gf.stem
        emb_file = emb_dir / f"{repo_name}.npz"

        # Resume support
        if emb_file.exists() and not args.force:
            skipped += 1
            continue

        # Load nodes
        try:
            nodes = load_graph(gf)
        except Exception as e:
            print(f"  [{i+1}/{len(graph_files)}] {repo_name}: FAILED to load: {e}")
            failed += 1
            continue

        if not nodes:
            print(f"  [{i+1}/{len(graph_files)}] {repo_name}: empty, skipping")
            continue

        n_nodes = len(nodes)
        n_comments = sum(1 for n in nodes if n.get("comment"))
        total_nodes += n_nodes
        total_with_comment += n_comments

        # Compute embeddings
        t0 = time.time()
        try:
            ids, embeddings = compute_repo_embeddings(nodes, model, batch_size=args.batch_size)
        except Exception as e:
            print(f"  [{i+1}/{len(graph_files)}] {repo_name}: FAILED to encode: {e}")
            failed += 1
            del nodes
            gc.collect()
            continue

        elapsed = time.time() - t0

        # Save as compressed npz
        np.savez_compressed(emb_file, ids=ids, embeddings=embeddings)

        print(f"  [{i+1}/{len(graph_files)}] {repo_name}: {n_nodes} nodes "
              f"({n_comments} with comments), {elapsed:.1f}s, "
              f"{emb_file.stat().st_size / 1024 / 1024:.1f} MB")

        del nodes, ids, embeddings
        gc.collect()
        processed += 1

    wall_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {wall_time:.0f}s")
    print(f"  Processed: {processed}, Skipped (existing): {skipped}, Failed: {failed}")
    print(f"  Total nodes embedded: {total_nodes} ({total_with_comment} with comments)")
    print(f"  Embeddings saved to: {emb_dir}")


if __name__ == "__main__":
    main()
