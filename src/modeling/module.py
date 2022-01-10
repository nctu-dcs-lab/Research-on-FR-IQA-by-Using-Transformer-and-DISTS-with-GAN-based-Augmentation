import torch
import torch.nn as nn

from src.modeling.backbone import InceptionResNetV2Backbone
from src.modeling.feature_projection import SingleFeatureProjection, MixedFeatureProjection
from src.modeling.transformer import Transformer, MLPHead


class Generator(nn.Module):
    def __init__(self, img_shape=(3, 192, 192), latent_dim=100):
        super().__init__()

        img_channels, self.img_height, self.img_wide = img_shape

        self.latent_encode = nn.Linear(latent_dim, self.img_height * self.img_wide)
        self.quality_encode = nn.Linear(1, self.img_height * self.img_wide)
        self.distort_encode = nn.Linear(1, self.img_height * self.img_wide)

        self.down1 = UNetDown(img_channels + 3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, img_channels, (3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, img, noise, quality, distort):
        noise = self.latent_encode(noise).view(noise.size(0), 1, self.img_height, self.img_wide)
        quality = self.quality_encode(quality).view(quality.size(0), 1, self.img_height, self.img_wide)
        distort = self.distort_encode(distort).view(distort.size(0), 1, self.img_height, self.img_wide)

        d1 = self.down1(torch.cat((img, noise, distort, quality), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, (3, 3), stride=(2, 2), padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, (3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.avgpool(feat).view(feat.size(0), -1)
        return self.classifer(feat)


class Evaluator(nn.Module):
    def __init__(self, cfg, num_pos, input_dim):
        super().__init__()

        if cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'mixed':
            self.feat_proj = MixedFeatureProjection(num_pos=num_pos,
                                                    input_dim=input_dim,
                                                    hidden_dim=cfg.MODEL.TRANSFORMER_DIM)
        else:
            self.feat_proj = SingleFeatureProjection(num_pos=num_pos,
                                                     input_dim=input_dim,
                                                     hidden_dim=cfg.MODEL.TRANSFORMER_DIM)

        self.transformer = Transformer(
            d_model=cfg.MODEL.TRANSFORMER_DIM,
            nhead=cfg.MODEL.MHA_NUM_HEADS,
            num_encoder_layers=cfg.MODEL.TRANSFORMER_LAYERS,
            num_decoder_layers=cfg.MODEL.TRANSFORMER_LAYERS,
            dim_feedforward=cfg.MODEL.FEAT_DIM
        )
        self.mlp_head = MLPHead(in_dim=cfg.MODEL.TRANSFORMER_DIM, hidden_dim=cfg.MODEL.HEAD_DIM)

    def forward(self, ref_feat, dist_feat):
        diff_feat = tuple(map(lambda i, j: i - j, ref_feat, dist_feat))

        ref_proj_feat = self.feat_proj(ref_feat)
        diff_proj_feat = self.feat_proj(diff_feat)

        return self.mlp_head(self.transformer(diff_proj_feat, ref_proj_feat)[0])


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=116, hidden_dim=512):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, feat):
        feat = self.avgpool(feat).view(feat.size(0), -1)
        return self.classifer(feat)


class MultiTask(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.MODEL.FEAT_EXTRACTOR_LEVEL in ['low', 'medium', 'high', 'mixed']

        hyperparameter = {
            'low': {'last_feat_dim': 320, 'feat_dim': 320 * 6, 'num_pos': 21 * 21},
            'medium': {'last_feat_dim': 1088, 'feat_dim': 1088 * 6, 'num_pos': 10 * 10},
            'high': {'last_feat_dim': 2080, 'feat_dim': 2080 * 6, 'num_pos': 4 * 4},
            'mixed': {
                'last_feat_dim': 2080,
                'feat_dim': (320 * 6, 1088 * 6, 2080 * 6),
                'num_pos': 21 * 21 + 10 * 10 + 4 * 4}
        }

        self.feat_extraction = InceptionResNetV2Backbone(level=cfg.MODEL.FEAT_EXTRACTOR_LEVEL)

        self.discriminator = Discriminator(input_dim=hyperparameter[cfg.MODEL.FEAT_EXTRACTOR_LEVEL]['last_feat_dim'])
        self.classifier = Classifier(input_dim=hyperparameter[cfg.MODEL.FEAT_EXTRACTOR_LEVEL]['last_feat_dim'])
        self.evaluator = Evaluator(cfg,
                                   num_pos=hyperparameter[cfg.MODEL.FEAT_EXTRACTOR_LEVEL]['num_pos'],
                                   input_dim=hyperparameter[cfg.MODEL.FEAT_EXTRACTOR_LEVEL]['feat_dim'])

    def forward(self, ref_img, dist_img):
        ref_feat = self.feat_extraction(ref_img)
        dist_feat = self.feat_extraction(dist_img)
        return self.discriminator(dist_feat[-1]).view(-1), self.classifier(dist_feat[-1]), self.evaluator(ref_feat,
                                                                                                          dist_feat)
