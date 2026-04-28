"""Bundle metadata + vocabularies for Java-side loading."""

import json
from pathlib import Path
from datetime import datetime

from ..model.graph_mae import EDGE_TYPES
from ..graph.features import (
    NODE_FEATURE_DIM, TEXT_EMBEDDING_DIM, METRIC_VECTOR_DIM,
    TYPE_ONE_HOT_DIM, LANGUAGE_ONE_HOT_DIM, COMPONENT_TYPES, LANGUAGES,
)


def export_metadata(output_dir: str = "artifacts/models"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schemaVersion": 1,
        "createdAt": datetime.utcnow().isoformat(),
        "edgeTypes": EDGE_TYPES,
        "nodeFeatureDim": NODE_FEATURE_DIM,
        "textEmbeddingDim": TEXT_EMBEDDING_DIM,
        "metricVectorDim": METRIC_VECTOR_DIM,
        "typeOneHotDim": TYPE_ONE_HOT_DIM,
        "languageOneHotDim": LANGUAGE_ONE_HOT_DIM,
        "componentTypes": COMPONENT_TYPES,
        "languages": LANGUAGES,
    }

    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to {output / 'metadata.json'}")

    # Placeholder metric normalizer
    normalizer = {
        "java": {
            "wmc": {"mean": 5.0, "std": 8.0},
            "dit": {"mean": 1.5, "std": 1.0},
            "noc": {"mean": 2.0, "std": 3.0},
            "ac": {"mean": 3.0, "std": 5.0},
            "ec": {"mean": 4.0, "std": 6.0},
            "encapsulation": {"mean": 0.8, "std": 0.3},
        },
        "python": {
            "wmc": {"mean": 4.0, "std": 7.0},
            "dit": {"mean": 1.2, "std": 0.8},
            "noc": {"mean": 1.5, "std": 2.5},
            "ac": {"mean": 2.5, "std": 4.0},
            "ec": {"mean": 3.5, "std": 5.0},
            "encapsulation": {"mean": 0.7, "std": 0.3},
        },
        "typescript": {
            "wmc": {"mean": 4.5, "std": 7.5},
            "dit": {"mean": 1.3, "std": 0.9},
            "noc": {"mean": 1.8, "std": 2.8},
            "ac": {"mean": 2.8, "std": 4.5},
            "ec": {"mean": 3.8, "std": 5.5},
            "encapsulation": {"mean": 0.75, "std": 0.3},
        },
    }

    with open(output / "metric_normalizer.json", "w") as f:
        json.dump(normalizer, f, indent=2)
    print(f"Exported metric normalizer to {output / 'metric_normalizer.json'}")


if __name__ == "__main__":
    export_metadata()
