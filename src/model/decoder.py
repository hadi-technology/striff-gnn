"""Edge reconstruction decoder for masked autoencoder training."""

import torch
import torch.nn as nn


class EdgeReconstructionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_edge_types: int = 5):
        super().__init__()
        self.edge_predictors = nn.ModuleDict()
        for i in range(num_edge_types):
            self.edge_predictors[str(i)] = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, node_embeddings, src_idx, tgt_idx, edge_type_idx):
        src_emb = node_embeddings[src_idx]
        tgt_emb = node_embeddings[tgt_idx]
        pair_feat = torch.cat([src_emb, tgt_emb], dim=-1)
        predictor = self.edge_predictors[str(edge_type_idx)]
        return predictor(pair_feat).squeeze(-1)
