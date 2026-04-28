"""Full GraphMAE-style masked autoencoder for architectural graph learning."""

import torch
import torch.nn as nn

from .encoder import ArchGNEncoder
from .decoder import EdgeReconstructionHead

EDGE_TYPES = ["EXTENSION", "IMPLEMENTATION", "COMPOSITION", "AGGREGATION", "ASSOCIATION"]


class ArchGraphMAE(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 404,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        edge_type_tuples = [("node", et, "node") for et in EDGE_TYPES]
        self.encoder = ArchGNEncoder(
            node_feat_dim=node_feat_dim,
            edge_types=edge_type_tuples,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.decoder = EdgeReconstructionHead(
            hidden_dim=hidden_dim,
            num_edge_types=len(EDGE_TYPES),
        )

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def predict_edges(self, node_embeddings, src_idx, tgt_idx, edge_type_idx):
        return self.decoder(node_embeddings, src_idx, tgt_idx, edge_type_idx)

    def forward(self, x_dict, edge_index_dict, query_pairs):
        h = self.encode(x_dict, edge_index_dict)
        outputs = {}
        for et_key, (src_idx, tgt_idx) in query_pairs.items():
            et_idx = EDGE_TYPES.index(et_key) if et_key in EDGE_TYPES else 0
            outputs[et_key] = self.predict_edges(h, src_idx, tgt_idx, et_idx)
        return outputs
