"""Loss functions for edge reconstruction training."""

import torch
import torch.nn as nn


def compute_edge_reconstruction_loss(predictions, labels):
    """Binary cross-entropy loss for edge existence prediction."""
    return nn.functional.binary_cross_entropy_with_logits(predictions, labels.float())


def compute_loss(predictions_dict, labels_dict):
    """Compute total loss across all edge types."""
    total_loss = torch.tensor(0.0)
    count = 0
    for key in predictions_dict:
        if key in labels_dict:
            preds = predictions_dict[key]
            labels = labels_dict[key]
            if preds.numel() > 0:
                total_loss = total_loss + compute_edge_reconstruction_loss(preds, labels)
                count += 1
    return total_loss / max(count, 1)
