"""Node and edge feature engineering for the GNN."""

import numpy as np
from typing import Optional

# Constants matching the Java side (FeatureBuilder)
TEXT_EMBEDDING_DIM = 384
METRIC_VECTOR_DIM = 9
TYPE_ONE_HOT_DIM = 7
LANGUAGE_ONE_HOT_DIM = 3
IS_SYNTHETIC_DIM = 1
NODE_FEATURE_DIM = TEXT_EMBEDDING_DIM + METRIC_VECTOR_DIM + TYPE_ONE_HOT_DIM + LANGUAGE_ONE_HOT_DIM + IS_SYNTHETIC_DIM  # 404

COMPONENT_TYPES = ["CLASS", "INTERFACE", "ENUM", "METHOD", "FIELD", "ANNOTATION", "OTHER"]
LANGUAGES = ["java", "python", "typescript"]
EDGE_TYPES = ["EXTENSION", "IMPLEMENTATION", "COMPOSITION", "AGGREGATION", "ASSOCIATION"]


def build_type_one_hot(component_type: str) -> np.ndarray:
    idx = COMPONENT_TYPES.index(component_type) if component_type in COMPONENT_TYPES else len(COMPONENT_TYPES) - 1
    vec = np.zeros(TYPE_ONE_HOT_DIM, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def build_language_one_hot(language: str) -> np.ndarray:
    idx = LANGUAGES.index(language.lower()) if language.lower() in LANGUAGES else 0
    vec = np.zeros(LANGUAGE_ONE_HOT_DIM, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def build_metric_vector(metrics: Optional[dict], cyclo: int = 0, refs_count: int = 0,
                        children_count: int = 0) -> np.ndarray:
    vec = np.zeros(METRIC_VECTOR_DIM, dtype=np.float32)
    if metrics:
        vec[0] = metrics.get("wmc", 0)
        vec[1] = metrics.get("dit", 0)
        vec[2] = metrics.get("noc", 0)
        vec[3] = metrics.get("ac", 0)
        vec[4] = metrics.get("ec", 0)
        vec[5] = metrics.get("encapsulation", 0)
    vec[6] = cyclo
    vec[7] = refs_count
    vec[8] = children_count
    return vec


def build_edge_type_one_hot(edge_type: str) -> np.ndarray:
    vec = np.zeros(len(EDGE_TYPES), dtype=np.float32)
    idx = EDGE_TYPES.index(edge_type) if edge_type in EDGE_TYPES else len(EDGE_TYPES) - 1
    vec[idx] = 1.0
    return vec
