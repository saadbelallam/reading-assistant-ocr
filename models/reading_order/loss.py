"""
Loss Function
"""

import torch
import torch.nn as nn

def weighted_bce_loss(logits, targets, pos_weight=1.0):
    # logits: raw model outputs, targets: 0/1 labels
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))
    return loss_fn(logits, targets.float())