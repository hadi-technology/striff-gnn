"""GNN encoder using HGTConv for heterogeneous graph encoding."""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class ArchGNEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_types: list[tuple],
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.input_proj = Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        metadata = (["node"], edge_types)
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, num_heads))

    def forward(self, x_dict, edge_index_dict):
        h = {"node": self.input_proj(x_dict["node"])}
        num_nodes = x_dict["node"].size(0)

        for conv in self.convs:
            # Pad missing edge types with self-loops to avoid HGTConv crash
            padded = dict(edge_index_dict)
            for et in self.edge_types:
                if et not in padded:
                    # Add a single self-loop at node 0
                    padded[et] = torch.tensor([[0], [0]], dtype=torch.long)
            h = conv(h, padded)
            h = {k: torch.relu(v) for k, v in h.items()}
        return h["node"]
