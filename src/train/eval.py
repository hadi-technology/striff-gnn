"""Evaluation on held-out repos with proper edge masking and hard negatives."""

import torch
import random
from torch_geometric.data import HeteroData
from sklearn.metrics import roc_auc_score
import numpy as np

from ..model.graph_mae import ArchGraphMAE, EDGE_TYPES


def _build_adjacency(data: HeteroData) -> set[tuple[int, int]]:
    """Build set of all (src, tgt) edges across all types."""
    edges = set()
    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key not in data.edge_types:
            continue
        ei = data[key].edge_index
        for s, t in zip(ei[0].tolist(), ei[1].tolist()):
            edges.add((s, t))
    return edges


def _sample_hard_negatives(
    data: HeteroData,
    num_neg: int,
    existing_edges: set[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample hard negatives: 2-hop neighbors not directly connected."""
    num_nodes = data["node"].x.size(0)
    if num_nodes < 2:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    neighbors = [set() for _ in range(num_nodes)]
    for et in EDGE_TYPES:
        key = ("node", et, "node")
        if key not in data.edge_types:
            continue
        ei = data[key].edge_index
        for s, t in zip(ei[0].tolist(), ei[1].tolist()):
            neighbors[s].add(t)
            neighbors[t].add(s)

    two_hop = [set() for _ in range(num_nodes)]
    for i in range(num_nodes):
        two_hop[i] = set(neighbors[i])
        for n in neighbors[i]:
            two_hop[i].update(neighbors[n])

    neg_src, neg_tgt = [], []
    attempts = 0
    while len(neg_src) < num_neg and attempts < num_neg * 10:
        s = random.randint(0, num_nodes - 1)
        if not two_hop[s]:
            attempts += 1
            continue
        candidates = list(two_hop[s] - neighbors[s] - {s})
        if not candidates:
            candidates = [j for j in range(num_nodes)
                          if j != s and (s, j) not in existing_edges]
        if candidates:
            t = random.choice(candidates)
            neg_src.append(s)
            neg_tgt.append(t)
        attempts += 1

    if not neg_src:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    return torch.tensor(neg_src, dtype=torch.long), torch.tensor(neg_tgt, dtype=torch.long)


def evaluate(model: ArchGraphMAE, data_list: list[HeteroData],
             device: str = "cpu", hard_negatives: bool = True) -> dict:
    """Evaluate by masking edges before encoding (no information leakage).

    For each validation graph and each edge type:
    1. Mask 50% of that edge type's edges
    2. Encode the graph WITHOUT those masked edges
    3. Predict whether the masked edges exist
    4. Use hard negatives as counter-examples
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            existing_edges = _build_adjacency(data) if hard_negatives else set()

            for eval_et_idx, eval_et in enumerate(EDGE_TYPES):
                eval_key = ("node", eval_et, "node")
                if eval_key not in data.edge_types:
                    continue

                full_ei = data[eval_key].edge_index
                if full_ei.size(1) == 0:
                    continue

                # Mask 50% of this edge type
                num_edges = full_ei.size(1)
                num_mask = max(1, num_edges // 2)
                mask_indices = set(random.sample(
                    range(num_edges), min(num_mask, num_edges)))

                # Build edge_index_dict WITHOUT the masked edges
                edge_index_dict = {}
                for et in EDGE_TYPES:
                    key = ("node", et, "node")
                    if key not in data.edge_types:
                        continue
                    if key == eval_key:
                        keep = [i for i in range(data[key].edge_index.size(1))
                                if i not in mask_indices]
                        if keep:
                            edge_index_dict[key] = data[key].edge_index[:, keep]
                    else:
                        edge_index_dict[key] = data[key].edge_index

                # Encode with masked graph
                x_dict = {"node": data["node"].x}
                h = model.encode(x_dict, edge_index_dict)

                # Positive: the masked edges
                pos_indices = list(mask_indices)
                pos_src = full_ei[0, pos_indices]
                pos_tgt = full_ei[1, pos_indices]
                if pos_src.numel() > 0:
                    preds = model.predict_edges(h, pos_src, pos_tgt, eval_et_idx)
                    all_scores.extend(torch.sigmoid(preds).cpu().numpy().tolist())
                    all_labels.extend([1.0] * pos_src.numel())

                # Hard negatives
                num_neg = min(pos_src.numel(), 200)
                if num_neg > 0 and hard_negatives:
                    neg_src, neg_tgt = _sample_hard_negatives(
                        data, num_neg, existing_edges)
                    if neg_src.numel() > 0:
                        neg_preds = model.predict_edges(
                            h, neg_src, neg_tgt, eval_et_idx)
                        all_scores.extend(
                            torch.sigmoid(neg_preds).cpu().numpy().tolist())
                        all_labels.extend([0.0] * neg_src.numel())
                elif num_neg > 0:
                    num_nodes = data["node"].x.size(0)
                    neg_src = torch.randint(
                        0, num_nodes, (num_neg,), device=device)
                    neg_tgt = torch.randint(
                        0, num_nodes, (num_neg,), device=device)
                    neg_preds = model.predict_edges(
                        h, neg_src, neg_tgt, eval_et_idx)
                    all_scores.extend(
                        torch.sigmoid(neg_preds).cpu().numpy().tolist())
                    all_labels.extend([0.0] * num_neg)

    if not all_labels:
        return {"auc": 0.5, "count": 0}

    auc = roc_auc_score(all_labels, all_scores)
    return {"auc": auc, "count": len(all_labels)}
