"""
model architecture
"""

import torch
import torch.nn as nn

class SpatialBranch(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # concatenated coords of box_i and box_j
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, coords_i, coords_j):
        x = torch.cat([coords_i, coords_j], dim=-1)
        return self.mlp(x)

class ReadingOrderModel(nn.Module):
    def __init__(self, spatial_dim=4, hidden_dim=64, pos_weight=None):
        super().__init__()
        self.spatial_branch = SpatialBranch(input_dim=spatial_dim, hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1)
        )

        # optionally store pos_weight for loss
        self.pos_weight = pos_weight
    
    def forward(self, box_1, box_2):
        spatial = self.spatial_branch(box_1, box_2)
        x =torch.cat([spatial], dim=-1)
        return self.classifier(x).squeeze(-1)