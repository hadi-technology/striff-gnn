# Architecture

## Overview

striff-gnn trains a Graph Neural Network to learn representations of software architecture. The core model is **ArchGraphMAE** — a masked autoencoder that learns by predicting whether edges exist between nodes in a software dependency graph.

**What does that mean?** Source code has structure — classes inherit from other classes, methods reference other types, interfaces are implemented by concrete classes. We turn this structure into a graph, then train a neural network to understand which nodes should be connected and how.

## Pipeline

```
                         ┌──────────────────────────────────────────────┐
                         │           Training Pipeline                  │
                         │                                              │
  Source repos ──────▶ Clone ──▶ Parse (Clarpse) ──▶ Normalize ──▶   │
                                                                │     │
                                          Text embeddings (384d) ─┼─▶  │
                                          OOP metrics (9d)       │     │
                                          Type one-hot (7d)      ┼─▶  │
                                          Language one-hot (3d)  │     │
                                          Synthetic flag (1d)   ─┘    │
                                                                │     │
                                          404-dim node features ──▶   │
                                                                         │
                                          HeteroData graph ──────────▶ │
                                          (per-type edge indices)      │
                                                                │     │
                                          Subgraph sampling ─────────▶ │
                                          (BFS cluster + random)      │
                                                                         │
                         └──────────────────────────────────────────────┘
```

**[Clarpse](https://github.com/hadii-tech/clarpse)** is the source code parser. It reads Java, Python, and TypeScript source files and outputs a language-agnostic structural model — classes, methods, fields, and the relationships between them. You run it as a local HTTP server and this pipeline calls its API.

## Model: ArchGraphMAE

### Encoder (ArchGNEncoder)

Uses [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)'s `HGTConv` — the Heterogeneous Graph Transformer ([Hu et al., 2020](https://arxiv.org/abs/2003.01332)).

**Key properties:**
- 3 layers, 4 attention heads, 128-dim hidden state
- Type-specific projections: separate `k_rel`, `v_rel` weight matrices per edge type
- Per-edge-type attention priors (`p_rel`) that modulate attention scores
- Input projection from 404 → 128 dims, followed by 3 HGT conv layers with ReLU

**Why HGT, not a plain GCN?** Software graphs are inherently heterogeneous — inheritance edges carry different semantic meaning than dependency edges. A standard GCN treats all edges the same. HGT learns type-specific aggregation patterns, so it can learn that "extends" and "uses" should propagate information differently.

### Decoder (EdgeReconstructionHead)

5 independent MLP heads, one per edge type. Each takes `(src_embedding || tgt_embedding)` → predicts edge existence (binary logit).

```
src_emb (128) ─┐
               ├──▶ concat (256) ──▶ Linear(256, 128) ──▶ ReLU ──▶ Linear(128, 1) ──▶ logit
tgt_emb (128) ─┘
```

### Training objective

**Masked edge reconstruction** with hard negative sampling:

1. Sample a focal set of nodes via BFS from a random seed
2. Mask ~50% of outgoing edges from focal nodes
3. Encode the graph **without** the masked edges (no information leakage)
4. For each masked edge (positive), sample a 2-hop neighbor that isn't directly connected (hard negative)
5. Binary cross-entropy loss: can the model distinguish real edges from negatives?

This forces the encoder to learn structural patterns — the model must understand which node pairs should be connected and what type of relationship they share.

## The Heterogeneous Edge Tradeoff

There are two export paths, reflecting a genuine architectural tension:

### Path 1: Heterogeneous HGT (`src/export/manual_hgt.py`)

The full HGT encoder re-implemented with primitive tensor ops (`matmul`, `scatter_add`, `softmax`) so it can be exported to ONNX. The trained PyG weights transfer directly because the math is identical.

- **Inputs**: `node_features` (N×404) + 5 separate `edge_index_{type}` (2×E) tensors
- **Preserves**: type-specific attention, relation-specific transforms
- **Tradeoff**: More expressive, but the consuming application needs to pass per-type edge indices

### Path 2: Homogeneous GCN (`src/export/to_onnx.py`)

A distilled GCN scorer that collapses all 5 edge types into a single adjacency matrix:

- **Inputs**: `node_features` (N×404) + `adj_matrix` (N×N) with D^{-1/2}AD^{-1/2} normalization
- **Collapses**: All edge types set `adj[src][tgt] = 1.0` regardless of type — the model can't tell an inheritance edge from a dependency edge
- **Tradeoff**: Simpler deployment (one matrix input), but loses the heterogeneous signal

**Why both exist**: The graph builder in the production API preserves per-type edge indices precisely so that switching to the HGT model is a scorer-side change only. The collapsed GCN is simpler to deploy and more stable with limited training data — but the HGT path becomes the right choice when the corpus grows large enough for type-specific weights to generalize without overfitting.

### Where the deployed model sits

The current production model (loaded by the STRIFF API) uses the **homogeneous GCN** path. The HGT export is the upgrade path.

## Feature Engineering

The 404-dimensional node feature vector captures both semantic and structural information:

| Feature group | Dims | Source | Encoding |
|---|---|---|---|
| Text embedding | 384 | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on name + comment | Dense float |
| OOP metrics | 9 | [Clarpse](https://github.com/hadii-tech/clarpse) static analysis | Raw float |
| Component type | 7 | CLASS, INTERFACE, ENUM, METHOD, FIELD, ANNOTATION, OTHER | One-hot |
| Language | 3 | Java, Python, TypeScript | One-hot |
| Synthetic | 1 | Auto-generated module nodes for Python/TS | Binary |

### OOP metrics breakdown

| Index | Metric | Meaning |
|---|---|---|
| 0 | WMC | Weighted Methods per Class (complexity) |
| 1 | DIT | Depth of Inheritance Tree |
| 2 | NOC | Number of Children |
| 3 | AC | Afferent Coupling (how many others depend on this) |
| 4 | EC | Efferent Coupling (how many this depends on) |
| 5 | Encapsulation | Encapsulation score |
| 6 | Cyclo | Cyclomatic complexity |
| 7 | Refs count | Number of references |
| 8 | Children count | Direct children count |

## Graph Construction

### For Java

[Clarpse](https://github.com/hadii-tech/clarpse) parses Java source into a component model with typed references. The normalizer (`src/corpus/normalizer.py`) maps:
- `extension` → EXTENSION
- `implementation` → IMPLEMENTATION
- `simple` → ASSOCIATION
- Parent-child containment → COMPOSITION

### For Python/TypeScript

Clarpse handles these languages too, but Python and TypeScript lack Java's explicit class/package structure — a Python file can have loose functions and variables at the top level. The normalizer creates **synthetic MODULE nodes** that group top-level functions and fields under a parent representing the source file. This gives Python/TS a structural graph similar to Java's.

These synthetic nodes are marked with `synthetic: true` in the graph and flagged in the feature vector (dimension 403 = 1.0), so the model can learn to treat them differently.

## References

- Hu, Z. et al. (2020). [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332). *WWW 2020*.
- Schlichtkrull, M. et al. (2018). [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). *ESWC 2018*.
- Hou, Z. et al. (2022). [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803). *KDD 2022*.
