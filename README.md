# striff-gnn

Graph Neural Network for software architecture analysis. Trains a Heterogeneous Graph Transformer (HGT) encoder via masked edge reconstruction to learn structural representations of codebases across Java, Python, and TypeScript.

Built as part of [STRIFF](https://striff.io) — AI-powered code review.

## How it works

The pipeline turns source code into graphs, then trains a GNN to predict whether nodes should be connected:

```
Source code          Structural           Node features         GNN model
from open-source  ─▶ graph extraction ──▶ (text + metrics +  ──▶ Learn which
repos                via Clarpse            type + language)       nodes connect
                                            = 404 dims per node    and how
```

The model learns a self-supervised task: we hide some edges in the graph, encode what remains, then ask the model to predict which edges we hid. This forces it to learn the structural patterns in real software.

**What the graph looks like:**
- **Nodes** = classes, interfaces, enums, methods, fields
- **Edges** = relationships between them:

| Edge type | Meaning | Example |
|-----------|---------|---------|
| EXTENSION | Class extends another class | `ArrayList` → `AbstractList` |
| IMPLEMENTATION | Class implements an interface | `ArrayList` → `List` |
| COMPOSITION | Strong ownership (has-a) | `Car` → `Engine` |
| AGGREGATION | Weak ownership (part-of) | `Department` → `Employee` |
| ASSOCIATION | General reference/usage | `Order` → `Customer` |

**What each node "knows"** (404 dimensions):
- **Text embedding** (384 dims): The node's name and comment, encoded by a sentence transformer (`all-MiniLM-L6-v2`)
- **OOP metrics** (9 dims): Complexity, coupling, inheritance depth, etc.
- **Component type** (7 dims): Is it a CLASS, INTERFACE, ENUM, METHOD, FIELD, ANNOTATION, or OTHER?
- **Language** (3 dims): Java, Python, or TypeScript?
- **Synthetic** (1 dim): Is this an auto-generated grouping node? (used for Python/TS files)

## Quick start

```bash
# Install
pip install -e .

# Verify everything works (no data or external services needed)
python scripts/smoke_test.py
```

You should see `SMOKE TEST PASSED` after a few seconds.

## Training pipeline

The full pipeline trains on real open-source repos. It has 4 stages:

### Prerequisites

1. **[Clarpse](https://github.com/hadii-tech/clarpse)** — a multi-language source code parser that extracts structural models (classes, methods, relationships) from Java, Python, and TypeScript code. It runs as a local HTTP server:
   ```bash
   # Clone and build (requires Java 17 + Maven)
   git clone https://github.com/hadii-tech/clarpse.git
   cd clarpse
   mvn clean package assembly:single

   # Start the API server
   java -cp target/clarpse-*.jar com.hadi.clarpse.server.ClarpseServer
   # Now listening on http://localhost:8080

   # Verify it's running:
   curl http://localhost:8080/health
   ```

2. **GitHub token** (optional, avoids rate limits when cloning repos):
   ```bash
   cp .env.example .env
   # Edit .env and add your token
   ```

All data is stored in `./data/` by default. Change this by setting `STRIFF_DATA_DIR` in `.env`.

### Stage 1: Build corpus

Clone the repos listed in `config/repos.yaml`, parse each with Clarpse, and convert to graph format:

```bash
python scripts/build_corpus.py
```

This creates:
- `data/corpus/repos/` — shallow-cloned source repos
- `data/corpus/graphs/` — normalized graph JSON files (see [docs/data-format.md](docs/data-format.md))
- `data/corpus/pt/` — [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (PyG) datasets

### Stage 2: Compute text embeddings

Generate 384-dim sentence embeddings for every node (using the node's name + doc comment):

```bash
python scripts/compute_embeddings.py
```

Creates `data/corpus/embeddings/{repo_name}.npz`. This downloads `all-MiniLM-L6-v2` on first run (~90 MB).

### Stage 3: Build training dataset

Sample subgraphs from each repo for training:

```bash
python scripts/build_dataset.py
```

Creates `data/dataset/{repo_name}_clustered_{i}.pt` (BFS-clustered subgraphs) and `data/dataset/{repo_name}_random_{i}.pt`.

### Stage 4: Train

```bash
python scripts/train.py
```

Training is resumable — if interrupted, it continues from the last checkpoint. Creates:
- `data/models/best_model.pt` — best model by validation AUC
- `data/models/final_model.pt` — model after the last epoch

## Model architecture

**ArchGraphMAE** has two parts:

1. **Encoder** — A 3-layer [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) (HGT) with 4 attention heads. HGT is a GNN architecture that learns *different* aggregation patterns for different edge types — so it treats inheritance edges differently from dependency edges. Hidden dimension is 128.

2. **Decoder** — 5 independent MLP heads (one per edge type). Each takes two node embeddings, concatenates them, and predicts whether that edge should exist.

**Training**: We mask ~50% of edges from a set of focal nodes, encode the graph *without* those edges, then ask the decoder to predict which ones were real. Hard negatives come from 2-hop neighbors that aren't directly connected. Loss is binary cross-entropy averaged across edge types.

**Evaluation**: AUC-ROC on held-out repos, with 50% edge masking per type and hard negatives.

See [docs/architecture.md](docs/architecture.md) for a deeper dive, including the tradeoff between the heterogeneous and homogeneous export paths.

## Export

### HGT encoder (recommended)

Exports the trained HGT encoder to ONNX:

```bash
python -c "from src.export.manual_hgt import export_encoder_onnx; \
  export_encoder_onnx('data/models/best_model.pt', 'data/models')"
```

Takes: `node_features` (N x 404) + 5 per-type edge index tensors.
Returns: `node_embeddings` (N x 128).

### GCN scorer (legacy)

A simpler model that collapses all edge types into one adjacency matrix:

```bash
python scripts/export.py
```

Takes: `node_features` (N x 404) + `adj_matrix` (N x N).
Returns: `anomaly_scores` (N,).

### Text encoder

Export the sentence transformer to ONNX for self-contained deployment:

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

Lists 60 open-source repositories (20 per language) used as training data. Add your own:

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
│   ├── model/              # ArchGraphMAE model
│   │   ├── encoder.py          # HGT encoder (uses PyG HGTConv)
│   │   ├── decoder.py          # Per-edge-type reconstruction head
│   │   └── graph_mae.py        # Full masked autoencoder
│   ├── graph/              # Graph construction and features
│   │   ├── features.py         # 404-dim feature builder
│   │   ├── dataset.py          # Graph → PyG HeteroData conversion
│   │   └── sampler.py          # Subgraph sampling (BFS + random)
│   ├── corpus/             # Data ingestion
│   │   ├── cloner.py           # Git repo cloning
│   │   ├── parser_client.py    # Clarpse API client
│   │   ├── normalizer.py       # Raw Clarpse output → normalized graph
│   │   └── source_parser.py    # Regex-based fallback parser
│   ├── train/              # Training and evaluation
│   │   ├── train.py            # Main training loop
│   │   ├── losses.py           # BCE loss
│   │   └── eval.py             # AUC evaluation with hard negatives
│   ├── export/             # ONNX export
│   │   ├── manual_hgt.py       # ONNX-exportable HGT (primitive ops)
│   │   ├── to_onnx.py          # Homogeneous GCN scorer export
│   │   └── metadata.py         # Model metadata bundling
│   └── tools/              # Debug utilities
├── config/                 # Training configs and repo lists
├── scripts/                # Pipeline entry points
├── docs/
│   ├── architecture.md         # Architecture deep-dive
│   └── data-format.md          # Graph format specification
└── pyproject.toml
```

## Citation

```bibtex
@software{striff-gnn,
  title = {striff-gnn: Graph Neural Networks for Software Architecture Analysis},
  author = {Muntazir Fadhel},
  url = {https://github.com/hadi-technology/striff-gnn},
  year = {2026}
}
```

## License

[MIT](LICENSE)
