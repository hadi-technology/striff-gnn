"""Main training loop for the ArchGraphMAE model.

Supports checkpointing: resumes from the latest checkpoint if one exists.
Saves a checkpoint after each epoch.
Streams datasets from disk one at a time to limit memory usage.
"""

import os
import gc
import yaml
import torch
import torch.optim as optim
from pathlib import Path

from ..model.graph_mae import ArchGraphMAE, EDGE_TYPES
from ..graph.dataset import load_dataset
from ..graph.sampler import sample_clustered
from .losses import compute_loss
from .eval import evaluate


def _build_neighbors(data) -> dict[int, set[int]]:
    """Build adjacency map for hard negative sampling."""
    neighbors = {}
    num_nodes = data["node"].x.size(0)
    for i in range(num_nodes):
        neighbors[i] = set()

    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key not in data.edge_types:
            continue
        ei = data[key].edge_index
        for s, t in zip(ei[0].tolist(), ei[1].tolist()):
            neighbors.setdefault(s, set()).add(t)
            neighbors.setdefault(t, set()).add(s)
    return neighbors


def _sample_hard_neg_tgt(neighbors: dict, src_nodes: list[int],
                         num_nodes: int) -> list[int]:
    """For each src, pick a target in its 2-hop neighborhood but not directly connected."""
    import random
    result = []
    for s in src_nodes:
        direct = neighbors.get(s, set())
        two_hop = set(direct)
        for n in direct:
            two_hop.update(neighbors.get(n, set()))
        # Candidates: in 2-hop but not direct neighbor, not self
        candidates = list(two_hop - direct - {s})
        if not candidates:
            # Fallback: any non-neighbor
            candidates = [j for j in range(num_nodes) if j != s and j not in direct]
        if candidates:
            result.append(random.choice(candidates))
        else:
            result.append(random.randint(0, num_nodes - 1))
    return result


def mask_focal_edges(data, focal_nodes, mask_ratio=0.5):
    """Mask outgoing edges from focal nodes for training with hard negatives.

    Returns:
        query_pairs: dict of edge_type -> (src_tensor, tgt_tensor) with pos+neg
        labels: dict of edge_type -> label tensor (1 for pos, 0 for neg)
        masked_edge_indices: dict of edge_type -> set of (src, tgt) tuples that were masked
    """
    import random

    focal_list = list(focal_nodes)
    num_mask = max(1, int(len(focal_list) * mask_ratio))
    masked_nodes = set(random.sample(focal_list, min(num_mask, len(focal_list))))

    # Build neighbor map for hard negative sampling
    neighbors = _build_neighbors(data)
    num_nodes = data["node"].x.size(0)

    query_pairs = {}
    labels = {}
    masked_edge_indices = {}  # Track which edges were masked

    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key not in data.edge_types:
            continue
        edge_index = data[key].edge_index
        src, tgt = edge_index[0], edge_index[1]

        mask = torch.zeros(src.size(0), dtype=torch.bool)
        for i in range(src.size(0)):
            if src[i].item() in masked_nodes:
                mask[i] = True

        masked_src = src[mask]
        masked_tgt = tgt[mask]

        # Track masked edges
        masked_set = set()
        for s, t in zip(masked_src.tolist(), masked_tgt.tolist()):
            masked_set.add((s, t))
        masked_edge_indices[et] = masked_set

        if masked_src.numel() > 0:
            # Hard negatives: 2-hop neighbors that aren't directly connected
            src_list = masked_src.tolist()
            neg_tgt_list = _sample_hard_neg_tgt(neighbors, src_list, num_nodes)
            neg_tgt = torch.tensor(neg_tgt_list, dtype=torch.long)

            query_pairs[et] = (
                torch.cat([masked_src, masked_src]),
                torch.cat([masked_tgt, neg_tgt]),
            )
            labels[et] = torch.cat([torch.ones(masked_src.size(0)), torch.zeros(len(neg_tgt_list))])

    return query_pairs, labels, masked_edge_indices


def train(
    data_dir: str | None = None,
    config_path: str = "config/training.yaml",
    output_dir: str | None = None,
):
    base = os.environ.get("STRIFF_DATA_DIR", "./data")
    if data_dir is None:
        data_dir = os.path.join(base, "dataset")
    if output_dir is None:
        output_dir = os.path.join(base, "models")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    train_config = config["training"]

    device = "cpu"
    model = ArchGraphMAE(
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    start_epoch = 0
    best_auc = 0.0
    checkpoint_path = output_path / "checkpoint.pt"
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best_auc={best_auc:.4f}")
    else:
        print("No checkpoint found, starting fresh")

    # Discover dataset files — stream from disk instead of loading all at once
    data_files = sorted(Path(data_dir).glob("*.pt"))
    print(f"Found {len(data_files)} dataset files")

    if not data_files:
        print("ERROR: No datasets found. Run build_corpus.py first.")
        return

    # Use last few files as validation (load once, keep in memory)
    val_count = min(config["dataset"]["validation_repos"], len(data_files))
    val_files = data_files[-val_count:]
    train_files = data_files[:-val_count] if val_count < len(data_files) else data_files

    val_data = []
    for f in val_files:
        d = load_dataset(f)
        if d is not None and d["node"].x.size(0) > 0:
            val_data.append(d)
    print(f"Train files: {len(train_files)}, Val datasets: {len(val_data)}")

    for epoch in range(start_epoch, train_config["epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Stream training data: load one at a time
        for fpath in train_files:
            data = load_dataset(fpath)
            if data is None or data["node"].x.size(0) < 3:
                del data
                continue

            data = data.to(device)
            num_nodes = data["node"].x.size(0)

            # Sample focal nodes
            size = min(
                max(5, num_nodes // 4),
                config["dataset"]["max_focal_size"],
            )
            focal = sample_clustered(data, size)
            if len(focal) < 3:
                del data
                continue

            query_pairs, labels, masked_edges = mask_focal_edges(data, focal)
            if not query_pairs:
                del data
                continue

            # Build edge_index_dict with masked edges REMOVED (no leakage)
            x_dict = {"node": data["node"].x}
            edge_index_dict = {}
            for et in EDGE_TYPES:
                key = ("node", et, "node")
                if key in data.edge_types:
                    full_ei = data[key].edge_index
                    masked_set = masked_edges.get(et, set())
                    if not masked_set:
                        edge_index_dict[key] = full_ei
                    else:
                        # Remove masked edges
                        keep = []
                        for idx in range(full_ei.size(1)):
                            pair = (full_ei[0, idx].item(), full_ei[1, idx].item())
                            if pair not in masked_set:
                                keep.append(idx)
                        if keep:
                            edge_index_dict[key] = full_ei[:, keep]
                        # If all edges removed, skip this type entirely

            predictions = model(x_dict, edge_index_dict, query_pairs)
            loss = compute_loss(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Free memory
            del data, predictions, loss, query_pairs, labels
            gc.collect()

        avg_loss = total_loss / max(num_batches, 1)

        # Evaluate
        metrics = evaluate(model, val_data, device)

        print(f"Epoch {epoch + 1}/{train_config['epochs']}: "
              f"loss={avg_loss:.4f}, val_auc={metrics['auc']:.4f}")

        # Save best model
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), output_path / "best_model.pt")
            print(f"  New best AUC: {best_auc:.4f}")

        # Save checkpoint (for resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
        }, checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), output_path / "final_model.pt")
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Models saved to {output_path}")


if __name__ == "__main__":
    train()
