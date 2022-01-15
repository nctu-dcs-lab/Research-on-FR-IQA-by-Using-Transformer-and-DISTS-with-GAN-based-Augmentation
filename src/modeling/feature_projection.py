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


class MixedFeatureProjection(FeatureProjection):
    """
    Feature Projection for mixed level InceptionResNet V2 backbone
    """

    def __init__(self, num_pos, input_dims, hidden_dim=256):
        super(MixedFeatureProjection, self).__init__(num_pos, hidden_dim)

        self.low_level_flatten_conv2d = nn.Sequential(
            nn.Conv2d(input_dims[0], hidden_dim, 1),
            nn.Flatten(start_dim=2, end_dim=-1)
        )
        self.medium_level_flatten_conv2d = nn.Sequential(
            nn.Conv2d(input_dims[1], hidden_dim, 1),
            nn.Flatten(start_dim=2, end_dim=-1)
        )
        self.high_level_flatten_conv2d = nn.Sequential(
            nn.Conv2d(input_dims[2], hidden_dim, 1),
            nn.Flatten(start_dim=2, end_dim=-1)
        )

    def forward_feat(self, feats) -> torch.Tensor:
        low_level_embed = self.low_level_flatten_conv2d(torch.cat(feats[:6], 1))
        medium_level_embed = self.medium_level_flatten_conv2d(torch.cat(feats[6:12], 1))
        high_level_embed = self.high_level_flatten_conv2d(torch.cat(feats[12:], 1))

        return torch.cat((low_level_embed, medium_level_embed, high_level_embed), 2).permute(0, 2, 1)


class SeparateFeatureProjection(FeatureProjection):
    """
    Feature Projection for general case
    """

    def __init__(self, num_pos, input_dims, hidden_dim=256):
        super(SeparateFeatureProjection, self).__init__(num_pos, hidden_dim)

        self.parts = nn.ModuleList([nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 1),
                                                  nn.Flatten(start_dim=2, end_dim=-1)) for input_dim in input_dims])

    def forward_feat(self, feats) -> torch.Tensor:
        projections = []
        for feat, part in zip(feats, self.parts):
            projections.append(part(feat))

        return torch.cat(projections, 2).permute(0, 2, 1)
