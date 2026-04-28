"""Build PyG HeteroData datasets from normalized graph outputs."""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from torch_geometric.data import HeteroData

from .features import (
    NODE_FEATURE_DIM, TEXT_EMBEDDING_DIM, METRIC_VECTOR_DIM,
    TYPE_ONE_HOT_DIM, LANGUAGE_ONE_HOT_DIM, IS_SYNTHETIC_DIM, EDGE_TYPES,
    build_type_one_hot, build_language_one_hot, build_metric_vector,
    build_edge_type_one_hot,
)

# Cache for loaded embedding files: {repo_name: {node_id: np.array}}
_embeddings_cache: dict[str, dict[str, np.ndarray]] = {}


def load_text_embeddings(repo_name: str,
                         corpus_dir: Optional[str] = None) -> dict[str, np.ndarray]:
    """Load pre-computed text embeddings for a repo.

    Looks for {corpus_dir}/corpus/embeddings/{repo_name}.npz produced by
    scripts/compute_embeddings.py.  Results are cached in-memory so repeated
    calls for the same repo are free.

    Args:
        repo_name: stem of the graph JSON file (e.g. "java_guava")
        corpus_dir: root data directory. Falls back to STRIFF_DATA_DIR env var
                    then ./data.

    Returns:
        dict mapping node_id -> 384-dim float32 embedding.
        Empty dict if the embedding file does not exist.
    """
    if repo_name in _embeddings_cache:
        return _embeddings_cache[repo_name]

    base = corpus_dir or os.environ.get("STRIFF_DATA_DIR", "./data")
    emb_path = Path(base) / "corpus" / "embeddings" / f"{repo_name}.npz"

    if not emb_path.exists():
        _embeddings_cache[repo_name] = {}
        return {}

    data = np.load(emb_path, allow_pickle=False)
    ids = data["ids"]          # array of strings
    embeddings = data["embeddings"]  # (N, 384) float32

    mapping = {}
    for node_id, emb in zip(ids, embeddings):
        mapping[str(node_id)] = emb

    _embeddings_cache[repo_name] = mapping
    return mapping


def clear_embeddings_cache() -> None:
    """Clear the in-memory embeddings cache to free RAM."""
    _embeddings_cache.clear()


def graph_to_hetero_data(graph: dict, language: str = "java",
                         text_embeddings: Optional[dict] = None,
                         repo_name: Optional[str] = None) -> HeteroData:
    """Convert a normalized graph dict to a PyG HeteroData object.

    Text embeddings are resolved in this order:
      1. The *text_embeddings* dict argument (if provided).
      2. Pre-computed embeddings loaded from the corpus embeddings directory
         (requires *repo_name* to be set).
      3. Zeros (no embeddings available).
    """
    data = HeteroData()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    metrics = graph.get("metrics", {})

    if not nodes:
        return data

    # Build node index
    node_ids = [n["id"] for n in nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # Resolve text embeddings: explicit dict > auto-load from disk > zeros
    if text_embeddings is None and repo_name is not None:
        text_embeddings = load_text_embeddings(repo_name)

    # Build node features
    features = np.zeros((len(nodes), NODE_FEATURE_DIM), dtype=np.float32)
    for i, node in enumerate(nodes):
        offset = 0

        # Text embedding (from pre-computed sentence-transformers)
        if text_embeddings and node["id"] in text_embeddings:
            emb = text_embeddings[node["id"]]
            features[i, offset:offset + TEXT_EMBEDDING_DIM] = emb[:TEXT_EMBEDDING_DIM]
        offset += TEXT_EMBEDDING_DIM

        # Metric vector
        node_metrics = metrics.get(node["id"], {})
        mv = build_metric_vector(node_metrics)
        features[i, offset:offset + METRIC_VECTOR_DIM] = mv
        offset += METRIC_VECTOR_DIM

        # Type one-hot
        features[i, offset:offset + TYPE_ONE_HOT_DIM] = build_type_one_hot(node["type"])
        offset += TYPE_ONE_HOT_DIM

        # Language one-hot
        features[i, offset:offset + LANGUAGE_ONE_HOT_DIM] = build_language_one_hot(language)
        offset += LANGUAGE_ONE_HOT_DIM

        # Is synthetic flag
        features[i, offset] = 1.0 if node.get("synthetic", False) else 0.0

    data["node"].x = torch.from_numpy(features)

    # Build edge indices per type
    for edge_type in EDGE_TYPES:
        src_list = []
        tgt_list = []
        for edge in edges:
            if edge["type"] == edge_type:
                src_idx = node_index.get(edge["src"])
                tgt_idx = node_index.get(edge["tgt"])
                if src_idx is not None and tgt_idx is not None:
                    src_list.append(src_idx)
                    tgt_list.append(tgt_idx)
        if src_list:
            data["node", edge_type, "node"].edge_index = torch.tensor(
                [src_list, tgt_list], dtype=torch.long
            )

    # Store metadata
    data["node"].node_ids = node_ids
    return data


def save_dataset(data: HeteroData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_dataset(path: Path) -> Optional[HeteroData]:
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)
