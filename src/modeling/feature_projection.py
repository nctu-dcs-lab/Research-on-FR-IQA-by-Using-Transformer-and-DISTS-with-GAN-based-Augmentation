import torch
from torch import nn as nn


class FeatureProjection(nn.Module):
    def __init__(self, num_pos, input_dims, hidden_dim):
        super(FeatureProjection, self).__init__()

        self.slices = nn.ModuleList([nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 1),
                                                   nn.Flatten(start_dim=2, end_dim=-1)) for input_dim in input_dims])

        self.quality_embed = nn.Embedding(1, hidden_dim)
        self.position_embed = nn.Embedding(num_pos + 1, hidden_dim)

    def forward(self, feats):
        batch_size = feats[0].shape[0]

        quality_embedding_list = []
        for feat, flatten_conv2d in zip(feats, self.slices):
            quality_embedding_list.append(flatten_conv2d(feat))
        quality_embedding = torch.cat(quality_embedding_list, 1).permute(0, 2, 1)

        extra_quality_embedding = self.quality_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        quality_embedding = torch.cat((extra_quality_embedding, quality_embedding), 1)

        position_embedding = self.position_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        return quality_embedding + position_embedding
