import abc

import torch
from torch import nn as nn

from src.modeling.feature_projection import FeatureProjection
from src.modeling.transformer import Transformer, MLPHead


class Evaluator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Evaluator, self).__init__()

    @abc.abstractmethod
    def forward(self, feats1, feats2):
        return NotImplemented


class DISTS(Evaluator):
    def __init__(self, cfg):
        super(DISTS, self).__init__()

        self.channels = cfg.MODEL.BACKBONE.CHANNELS

        alpha = torch.zeros(1, sum(self.channels), 1, 1)
        beta = torch.zeros(1, sum(self.channels), 1, 1)

        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, feats1, feats2):
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        alpha = self.alpha.sigmoid()
        beta = self.beta.sigmoid()
        w_sum = alpha.sum() + beta.sum()
        alpha = torch.split(alpha / w_sum, self.channels, dim=1)
        beta = torch.split(beta / w_sum, self.channels, dim=1)

        for k in range(len(self.channels)):
            x_mean = feats1[k].mean([2, 3], keepdim=True)
            y_mean = feats2[k].mean([2, 3], keepdim=True)
            s1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * s1).sum(1, keepdim=True)

            x_var = ((feats1[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats2[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats1[k] * feats2[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            s2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * s2).sum(1, keepdim=True)

        return 1 - torch.squeeze(dist1 + dist2)


class IQT(Evaluator):
    def __init__(self, cfg):
        super(IQT, self).__init__()

        self.feat_proj = FeatureProjection(
            num_pos=cfg.MODEL.IQT.NUM_POS,
            input_dims=cfg.MODEL.BACKBONE.CHANNELS,
            hidden_dim=cfg.MODEL.IQT.TRANSFORMER_DIM
        )

        self.transformer = Transformer(
            d_model=cfg.MODEL.IQT.TRANSFORMER_DIM,
            nhead=cfg.MODEL.IQT.MHA_NUM_HEADS,
            num_encoder_layers=cfg.MODEL.IQT.TRANSFORMER_LAYERS,
            num_decoder_layers=cfg.MODEL.IQT.TRANSFORMER_LAYERS,
            dim_feedforward=cfg.MODEL.IQT.FEAT_DIM
        )
        self.mlp_head = MLPHead(in_dim=cfg.MODEL.IQT.TRANSFORMER_DIM, hidden_dim=cfg.MODEL.IQT.HEAD_DIM)

    def forward(self, ref_feat, dist_feat):
        diff_feat = tuple(map(lambda i, j: i - j, ref_feat, dist_feat))

        ref_proj_feat = self.feat_proj(ref_feat)
        diff_proj_feat = self.feat_proj(diff_feat)

        return self.mlp_head(self.transformer(diff_proj_feat, ref_proj_feat)[0])