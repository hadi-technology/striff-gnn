# CLAUDE.md

## Project overview
striff-gnn trains a Graph Neural Network (HGT-based masked autoencoder) for software architecture analysis. It extracts structural graphs from codebases, encodes them, and exports to ONNX for the production scorer.

## Key paths

- **This repo**: `/mnt/share/git/striff-gnn`
- **Production API (striff-api)**: `/mnt/share/git/striff-api`
- **Training data**: `./data/` (gitignored, lives on the NAS at `/mnt/share/striff-NeSy/`)

## Experiment → export → deploy workflow

After making model changes and retraining:

```bash
# 1. Train (resumable from checkpoint)
python scripts/train.py

# 2. Export ONNX model + metadata
python -c "from src.export.manual_hgt import export_encoder_onnx; export_encoder_onnx('data/models/best_model.pt', 'data/models')"
python -c "from src.export.metadata import export_metadata; export_metadata('data/models')"

# 3. Copy artifacts into striff-api resources
cp data/models/arch_scorer.onnx  /mnt/share/git/striff-api/src/main/resources/models/
cp data/models/metadata.json     /mnt/share/git/striff-api/src/main/resources/models/
cp data/models/metric_normalizer.json /mnt/share/git/striff-api/src/main/resources/models/
```

For the legacy homogeneous GCN scorer (currently deployed):
```bash
python scripts/export.py
```

For the text encoder:
```bash
python scripts/export_text_encoder_onnx.py
cp data/models/text_encoder.onnx /mnt/share/git/striff-api/src/main/resources/models/
```

## Currently deployed model

The striff-api **currently loads the homogeneous GCN** (`to_onnx.py` export) which takes:
- Input: `node_features` (N×404), `adj_matrix` (N×N)
- Output: `anomaly_scores` (N,)

The **HGT encoder** (`manual_hgt.py` export) is the upgrade path:
- Input: `node_features` (N×404) + 5 per-type `edge_index` tensors
- Output: `node_embeddings` (N×128)

Switching striff-api to the HGT model requires updating `OnnxArchitecturalScorer.java` and `GraphBuilder.java` to pass per-type edge indices instead of a collapsed adjacency matrix.

## Running

```bash
# Smoke test (no data needed)
python scripts/smoke_test.py

# Full pipeline
python scripts/build_corpus.py      # clone + parse repos
python scripts/compute_embeddings.py # sentence-transformer embeddings
python scripts/build_dataset.py      # sample subgraphs
python scripts/train.py              # train model
```

All scripts use `STRIFF_DATA_DIR` env var (default: `./data`).

## Architecture notes

- **Model**: ArchGraphMAE = HGT encoder (3 layers, 4 heads, 128-dim) + per-edge-type reconstruction decoder
- **Features**: 404-dim = text emb (384) + metrics (9) + type one-hot (7) + language one-hot (3) + synthetic flag (1)
- **Edge types**: EXTENSION, IMPLEMENTATION, COMPOSITION, AGGREGATION, ASSOCIATION
- **Training**: Masked edge reconstruction with hard negative sampling (2-hop neighbors)
- **Eval**: AUC-ROC on held-out repos with 50% edge masking
