"""
model architecture (dual branch)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

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
        # coords_i, coords_j: (batch_size, 4) -> x, y, w, h
        x = torch.cat([coords_i, coords_j], dim=-1)
        return self.mlp(x)

class SemanticBranch(nn.Module):
    def __init__(self, model_name="camembert-base", hidden_dim=64, freeze_transformer=True):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.pooler = nn.Linear(config.hidden_size * 2, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, input_ids_i, attention_mask_i, input_ids_j, attention_mask_j):
        # get [CLS] embeddings for both regions
        out_i = self.transformer(input_ids=input_ids_i, attention_mask=attention_mask_i).last_hidden_state[:, 0, :]
        out_j = self.transformer(input_ids=input_ids_j, attention_mask=attention_mask_j).last_hidden_state[:, 0, :]
        x = torch.cat([out_i, out_j], dim=-1)
        return self.activation(self.pooler(x))

class ReadingOrderModel(nn.Module):
    def __init__(self, spatial_dim=4, hidden_dim=64, model_name="camembert-base", pos_weight=None):
        super().__init__()
        self.spatial_branch = SpatialBranch(input_dim=spatial_dim, hidden_dim=hidden_dim)
        self.semantic_branch = SemanticBranch(model_name=model_name, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # optionally store pos_weight for loss
        self.pos_weight = pos_weight

    def forward(self, coords_i, coords_j, input_ids_i, attention_mask_i, input_ids_j, attention_mask_j):
        spatial_feat = self.spatial_branch(coords_i, coords_j)
        semantic_feat = self.semantic_branch(input_ids_i, attention_mask_i, input_ids_j, attention_mask_j)
        x = torch.cat([spatial_feat, semantic_feat], dim=-1)
        logits = self.classifier(x).squeeze(-1)
        return logits