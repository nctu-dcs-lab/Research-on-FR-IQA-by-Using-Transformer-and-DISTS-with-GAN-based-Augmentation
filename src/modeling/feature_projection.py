import torch
from torch import nn as nn


class FeatureProjection(nn.Module):
    """
    A base class for feature projection
    """

    def __init__(self, num_pos, hidden_dim):
        super(FeatureProjection, self).__init__()

        self.quality_embed = nn.Embedding(1, hidden_dim)
        self.position_embed = nn.Embedding(num_pos + 1, hidden_dim)

    def forward_feat(self, feats) -> torch.Tensor:
        pass

    def forward(self, feats):
        batch_size = feats[0].shape[0]

        extra_quality_embedding = self.quality_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        quality_embedding = torch.cat((extra_quality_embedding, self.forward_feat(feats)), 1)

        position_embedding = self.position_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        return quality_embedding + position_embedding


class IQTFeatureProjection(FeatureProjection):
    """
    Feature Projection for IQT
    """

    def __init__(self, num_pos, input_dim, hidden_dim=256):
        super().__init__(num_pos, hidden_dim)

        self.flatten_conv2d = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            nn.Flatten(start_dim=2, end_dim=-1)
        )

    def forward_feat(self, feats):
        feat = torch.cat(feats, 1)
        return self.flatten_conv2d(feat).permute(0, 2, 1)
