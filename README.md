# striff-gnn

Graph Neural Network for software architecture analysis. Trains a Heterogeneous Graph Transformer (HGT) encoder via masked edge reconstruction to learn structural representations of codebases across Java, Python, and TypeScript.

Built as part of [STRIFF](https://striff.io) — AI-powered code review.

## How it works

```
Source repos ──▶ Clarpse parser ──▶ Normalized graphs ──▶ 404-dim features ──▶ HGT encoder ──▶ Edge reconstruction
                      │                    │                     │                    │
                      │              {nodes, edges,        text emb (384) +      ArchGraphMAE
                      │               metrics}             metrics (9) +         (HGT + decoder)
                      │                                    type (7) +
                      │                                    language (3) +
                      │                                    synthetic (1)
```

The model learns whether two nodes in a software graph should be connected, and what type of relationship connects them. This is trained as a self-supervised edge reconstruction task: mask some edges, encode the graph, then predict which masked edges exist.

**Edge types** (heterogeneous relations):
| Type | Meaning |
|------|---------|
| EXTENSION | Class extends another class |
| IMPLEMENTATION | Class implements an interface |
| COMPOSITION | Strong ownership (has-a) |
| AGGREGATION | Weak ownership (part-of) |
| ASSOCIATION | General reference/usage |

**Node features** (404 dimensions):
- Text embedding (384-dim) from `all-MiniLM-L6-v2` sentence transformer on name + comment
- OOP metrics (9-dim): WMC, DIT, NOC, AC, EC, encapsulation, cyclomatic complexity, refs count, children count
- Component type one-hot (7-dim): CLASS, INTERFACE, ENUM, METHOD, FIELD, ANNOTATION, OTHER
- Language one-hot (3-dim): Java, Python, TypeScript
- Synthetic flag (1-dim): marks auto-generated module nodes

## Quick start

```bash
# Install dependencies
pip install -e .

# Run the smoke test (no data or external services needed)
python scripts/smoke_test.py
```

## Training pipeline

The full pipeline has 4 stages. All data goes into `./data/` by default (configurable via `STRIFF_DATA_DIR`).

### Prerequisites

- **Clarpse server**: Docker container that parses source code into structural models
  ```bash
  docker run -p 8080:8080 clarpse-server
  ```
- **GitHub token** (optional, avoids rate limits when cloning): set `GITHUB_TOKEN` in `.env`
- Copy `.env.example` to `.env` and configure

### Stage 1: Build corpus

Clone repos from `config/repos.yaml` and parse them into normalized graph JSON + PyG datasets:

```bash
python scripts/build_corpus.py
```

Output: `data/corpus/repos/`, `data/corpus/graphs/`, `data/corpus/pt/`

### Stage 2: Compute text embeddings

Generate 384-dim sentence embeddings for all nodes using `all-MiniLM-L6-v2`:

```bash
python scripts/compute_embeddings.py
```

Output: `data/corpus/embeddings/{repo_name}.npz`

### Stage 3: Build training dataset

Sample subgraphs (clustered BFS + random) from each repo:

```bash
python scripts/build_dataset.py
```

Output: `data/dataset/{repo_name}_clustered_{i}.pt`, `data/dataset/{repo_name}_random_{i}.pt`

### Stage 4: Train

Train the ArchGraphMAE model. Resumable from checkpoint:

```bash
python scripts/train.py
```

Output: `data/models/best_model.pt`, `data/models/final_model.pt`

## Model architecture

**ArchGraphMAE** combines a Heterogeneous Graph Transformer encoder with an edge reconstruction decoder:

- **Encoder** (`ArchGNEncoder`): 3-layer HGTConv with 4 attention heads, hidden dim 128. Uses type-specific key/value transforms (`k_rel`, `v_rel`) and edge-type attention priors (`p_rel`) so the model learns different aggregation patterns for inheritance vs composition edges.

- **Decoder** (`EdgeReconstructionHead`): Per-edge-type MLP that takes concatenated (src, tgt) embeddings and predicts edge existence. 5 independent heads, one per relation type.

- **Training**: Masked edge reconstruction with hard negative sampling (2-hop neighbors that aren't directly connected). Binary cross-entropy loss averaged across edge types.

- **Evaluation**: AUC-ROC on held-out repos with 50% edge masking per type and hard negatives.

## Export

### Heterogeneous HGT encoder (recommended)

Exports the trained HGT encoder to ONNX using a manual reimplementation of HGTConv with primitive tensor ops:

```bash
python -c "from src.export.manual_hgt import export_encoder_onnx; export_encoder_onnx('data/models/best_model.pt', 'data/models')"
```

Inputs: `node_features` (N x 404) + 5 separate `edge_index_{type}` tensors.
Output: `node_embeddings` (N x 128).

### Homogeneous GCN scorer (legacy)

Collapsed GCN that uses a single adjacency matrix:

```bash
python scripts/export.py
```

Inputs: `node_features` (N x 404) + `adj_matrix` (N x N).
Output: `anomaly_scores` (N,).

See [docs/architecture.md](docs/architecture.md) for the design tradeoff between these two approaches.

### Text encoder

Export `all-MiniLM-L6-v2` to ONNX for self-contained deployment:

```bash
python scripts/export_text_encoder_onnx.py
```

## Configuration

### `config/training.yaml`

```yaml
model:
  hidden_dim: 128        # HGT hidden dimension
  num_layers: 3          # Number of HGT layers
  num_heads: 4           # Attention heads per layer

training:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01

dataset:
  clustered_samples: 20  # BFS-clustered subgraphs per repo
  random_samples: 5      # Random subgraphs per repo
  min_focal_size: 50     # Minimum subgraph node count
  max_focal_size: 500    # Maximum subgraph node count
  validation_repos: 5    # Number of repos held out for validation
```

### `config/repos.yaml`

Lists 60 open-source repositories (20 per language) used as training data. Edit to add/remove repos:

```yaml
java:
  - { url: https://github.com/spring-projects/spring-framework, ref: main }
python:
  - { url: https://github.com/django/django, ref: main }
typescript:
  - { url: https://github.com/microsoft/TypeScript, ref: main }
```

## Project structure

```
striff-gnn/
├── src/
│   ├── model/          # ArchGraphMAE model (encoder + decoder)
│   │   ├── encoder.py      # HGT-based encoder using PyG HGTConv
│   │   ├── decoder.py      # Per-edge-type reconstruction head
│   │   └── graph_mae.py    # Full masked autoencoder
│   ├── graph/          # Graph construction and feature engineering
│   │   ├── features.py     # 404-dim feature vectors
│   │   ├── dataset.py      # Graph → PyG HeteroData conversion
│   │   └── sampler.py      # Subgraph sampling (BFS + random)
│   ├── corpus/         # Data ingestion pipeline
│   │   ├── cloner.py       # Git repo cloning
│   │   ├── parser_client.py # Clarpse API client
│   │   ├── normalizer.py   # Raw parse → normalized graph
│   │   └── source_parser.py # Regex-based fallback parser
│   ├── train/          # Training and evaluation
│   │   ├── train.py        # Main training loop
│   │   ├── losses.py       # BCE loss for edge reconstruction
│   │   └── eval.py         # AUC evaluation with hard negatives
│   ├── export/         # ONNX export utilities
│   │   ├── manual_hgt.py   # ONNX-exportable HGT (primitive ops)
│   │   ├── to_onnx.py      # Homogeneous GCN scorer export
│   │   └── metadata.py     # Model metadata bundling
│   └── tools/          # Debugging utilities
├── config/             # Training configs and repo lists
├── scripts/            # Pipeline entry points
├── docs/
│   ├── architecture.md     # Architecture deep-dive
│   └── data-format.md      # Graph format specification
├── pyproject.toml
└── LICENSE            # MIT
```

## Links

- **STRIFF**: [striff.io](https://striff.io)
- **Clarpse**: Source code structural parser used for graph extraction

## Citation

If you use this in your research, please cite:

```bibtex
@software{striff-gnn,
  title = {striff-gnn: Graph Neural Networks for Software Architecture Analysis},
  author = {STRIFF},
  url = {https://github.com/striff-io/striff-gnn},
  year = {2025}
}
```

## License

[MIT](LICENSE)
