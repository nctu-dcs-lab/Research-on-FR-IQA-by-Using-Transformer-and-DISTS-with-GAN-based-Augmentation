import abc

import torch
from torch import nn as nn

from src.modeling.feature_projection import IQTFeatureProjection, SeparateFeatureProjection, MixedFeatureProjection
from src.modeling.transformer import Transformer, MLPHead


class Evaluator(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Evaluator, self).__init__()

    @abc.abstractmethod
    def forward(self, feats1, feats2):
        return NotImplemented


class DISTS(Evaluator):
    def __init__(self, backbone_channels):
        super(DISTS, self).__init__()

        self.channels = backbone_channels

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


class TransformerEvaluator(Evaluator):
    def __init__(self, cfg, backbone_channels, backbone_output_size):
        super(TransformerEvaluator, self).__init__()

        self.feat_proj = SeparateFeatureProjection(
            num_pos=sum(backbone_output_size),
            input_dims=backbone_channels,
            hidden_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM
        )

        self.transformer = Transformer(
            d_model=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM,
            nhead=cfg.MODEL.TRANSFORMER.MHA_NUM_HEADS,
            num_encoder_layers=cfg.MODEL.TRANSFORMER.TRANSFORMER_LAYERS,
            num_decoder_layers=cfg.MODEL.TRANSFORMER.TRANSFORMER_LAYERS,
            dim_feedforward=cfg.MODEL.TRANSFORMER.FEAT_DIM
        )
        self.mlp_head = MLPHead(in_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM, hidden_dim=cfg.MODEL.TRANSFORMER.HEAD_DIM)

    def forward(self, ref_feat, dist_feat):
        diff_feat = tuple(map(lambda i, j: i - j, ref_feat, dist_feat))

        ref_proj_feat = self.feat_proj(ref_feat)
        diff_proj_feat = self.feat_proj(diff_feat)

        return self.mlp_head(self.transformer(diff_proj_feat, ref_proj_feat)[0])


class IQT(TransformerEvaluator):
    def __init__(self, cfg, backbone_channels, backbone_output_size):
        super(IQT, self).__init__(cfg, backbone_channels, backbone_output_size)
        assert cfg.MODEL.BACKBONE.NAME == 'InceptionResNetV2'
        assert cfg.MODEL.BACKBONE.FEAT_LEVEL in ['low', 'medium', 'high', 'mixed']

        if cfg.MODEL.BACKBONE.FEAT_LEVEL == 'mixed':
            self.feat_proj = MixedFeatureProjection(
                num_pos=backbone_output_size[0] + backbone_output_size[6] + backbone_output_size[12],
                input_dims=(sum(backbone_channels[0:6]),
                            sum(backbone_channels[6:12]),
                            sum(backbone_channels[12:])),
                hidden_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM
            )
        else:
            self.feat_proj = IQTFeatureProjection(
                num_pos=backbone_output_size[0],
                input_dim=sum(backbone_channels),
                hidden_dim=cfg.MODEL.TRANSFORMER.TRANSFORMER_DIM
            )
