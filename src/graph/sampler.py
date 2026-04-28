"""Subgraph sampling strategies: clustered and random."""

import random
from collections import defaultdict, deque
from typing import Optional

import torch
from torch_geometric.data import HeteroData

from .features import EDGE_TYPES


def sample_clustered(data: HeteroData, size: int, seed: Optional[int] = None) -> set[int]:
    """Sample a clustered subgraph by BFS from a random seed node."""
    if seed is not None:
        random.seed(seed)

    num_nodes = data["node"].x.size(0)
    if num_nodes == 0:
        return set()

    start = random.randint(0, num_nodes - 1)
    visited = {start}
    frontier = deque([start])

    while frontier and len(visited) < size:
        node = frontier.popleft()
        for edge_type in EDGE_TYPES:
            key = ("node", edge_type, "node")
            if key in data.edge_types:
                edge_index = data[key].edge_index
                # Outgoing
                mask = edge_index[0] == node
                targets = edge_index[1][mask].tolist()
                for t in targets:
                    if len(visited) >= size:
                        break
                    if t not in visited:
                        visited.add(t)
                        frontier.append(t)
                # Incoming
                mask = edge_index[1] == node
                sources = edge_index[0][mask].tolist()
                for s in sources:
                    if len(visited) >= size:
                        break
                    if s not in visited:
                        visited.add(s)
                        frontier.append(s)

    return visited


def sample_random(data: HeteroData, size: int, seed: Optional[int] = None) -> set[int]:
    """Sample random nodes."""
    if seed is not None:
        random.seed(seed)

    num_nodes = data["node"].x.size(0)
    if num_nodes == 0:
        return set()

    size = min(size, num_nodes)
    return set(random.sample(range(num_nodes), size))


def extract_subgraph(data: HeteroData, focal_nodes: set[int]) -> HeteroData:
    """Extract a subgraph induced by the focal nodes."""
    sub = HeteroData()
    node_list = sorted(focal_nodes)
    old_to_new = {old: new for new, old in enumerate(node_list)}

    sub["node"].x = data["node"].x[node_list]
    if hasattr(data["node"], "node_ids"):
        ids = data["node"].node_ids
        sub["node"].node_ids = [ids[i] for i in node_list if i < len(ids)]

    for edge_type in EDGE_TYPES:
        key = ("node", edge_type, "node")
        if key in data.edge_types:
            edge_index = data[key].edge_index
            src, tgt = edge_index[0].tolist(), edge_index[1].tolist()
            filtered_src, filtered_tgt = [], []
            for s, t in zip(src, tgt):
                if s in focal_nodes and t in focal_nodes:
                    filtered_src.append(old_to_new[s])
                    filtered_tgt.append(old_to_new[t])
            if filtered_src:
                sub[key].edge_index = torch.tensor(
                    [filtered_src, filtered_tgt], dtype=torch.long
                )

    return sub
