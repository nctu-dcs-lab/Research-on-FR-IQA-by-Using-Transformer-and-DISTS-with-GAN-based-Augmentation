import copy

import timm
import torch
import torch.nn as nn


class FeatureExtractorInceptionResNetV2(nn.Module):
    def __init__(self, level='low'):
        super().__init__()

        inception_resnet_v2_pretrained = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2_pretrained_features = list(inception_resnet_v2_pretrained.children())

        if level == 'low':
            self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:8])
            self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][:2])
            self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][2:4])
            self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][4:6])
            self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][6:8])
            self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][8:10])

        elif level == 'medium':
            self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:10])
            self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][:4])
            self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][4:8])
            self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][8:12])
            self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][12:16])
            self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][16:])

        elif level == 'high':
            self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:12])
            self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][:2])
            self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][2:4])
            self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][4:6])
            self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][6:8])
            self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][8:],
                                        inception_resnet_v2_pretrained_features[13])

        # default using low level feature
        else:
            self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:8])
            self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][:2])
            self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][2:4])
            self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][4:6])
            self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][6:8])
            self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][8:10])

    def forward(self, x):
        feat0 = self.slice1(x)
        feat1 = self.slice2(feat0)
        feat2 = self.slice3(feat1)
        feat3 = self.slice4(feat2)
        feat4 = self.slice5(feat3)
        feat5 = self.slice6(feat4)

        return feat0, feat1, feat2, feat3, feat4, feat5


class MixedFeatureExtractorInceptionResNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        inception_resnet_v2_pretrained = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2_pretrained_features = list(inception_resnet_v2_pretrained.children())

        # low level feature block
        self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:8])
        self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][:2])
        self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][2:4])
        self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][4:6])
        self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][6:8])
        self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][8:10])

        # medium level feature block
        self.slice7 = nn.Sequential(*inception_resnet_v2_pretrained_features[9:10])
        self.slice8 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][:4])
        self.slice9 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][4:8])
        self.slice10 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][8:12])
        self.slice11 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][12:16])
        self.slice12 = nn.Sequential(*inception_resnet_v2_pretrained_features[10][16:])

        # high level feature block
        self.slice13 = nn.Sequential(*inception_resnet_v2_pretrained_features[11:12])
        self.slice14 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][:2])
        self.slice15 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][2:4])
        self.slice16 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][4:8])
        self.slice17 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][8:16])
        self.slice18 = nn.Sequential(*inception_resnet_v2_pretrained_features[12][8:],
                                     inception_resnet_v2_pretrained_features[13])

    def forward(self, x):
        # low level feature
        feat0 = self.slice1(x)
        feat1 = self.slice2(feat0)
        feat2 = self.slice3(feat1)
        feat3 = self.slice4(feat2)
        feat4 = self.slice5(feat3)
        feat5 = self.slice6(feat4)
        low_level_feat = (feat0, feat1, feat2, feat3, feat4, feat5)

        # medium level feature
        feat6 = self.slice7(feat5)
        feat7 = self.slice8(feat6)
        feat8 = self.slice9(feat7)
        feat9 = self.slice10(feat8)
        feat10 = self.slice11(feat9)
        feat11 = self.slice12(feat10)
        medium_level_feat = (feat6, feat7, feat8, feat9, feat10, feat11)

        # high level feature
        feat12 = self.slice13(feat11)
        feat13 = self.slice14(feat12)
        feat14 = self.slice15(feat13)
        feat15 = self.slice16(feat14)
        feat16 = self.slice17(feat15)
        feat17 = self.slice18(feat16)
        high_level_feat = (feat12, feat13, feat14, feat15, feat16, feat17)

        return low_level_feat + medium_level_feat + high_level_feat


class FeatureProjection(nn.Module):
    def __init__(self, num_pos=841, input_dim=1920, hidden_dim=256):
        super().__init__()

        self.flatten_conv2d = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            nn.Flatten(start_dim=2, end_dim=-1)
        )

        self.quality_embed = nn.Embedding(1, hidden_dim)
        self.position_embed = nn.Embedding(num_pos + 1, hidden_dim)

    def forward(self, feat):
        feat = torch.cat(feat, 1)
        batch_size = feat.shape[0]
        quality_embedding = self.flatten_conv2d(feat).permute(0, 2, 1)
        extra_quality_embedding = self.quality_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        quality_embedding = torch.cat((extra_quality_embedding, quality_embedding), 1)

        position_embedding = self.position_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        return quality_embedding + position_embedding


class MixedFeatureProjection(nn.Module):
    def __init__(self, num_pos=21 * 21 + 10 * 10 + 4 * 4, input_dims=(1920, 6528, 12480), hidden_dim=256):
        super().__init__()

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

        self.quality_embed = nn.Embedding(1, hidden_dim)
        self.position_embed = nn.Embedding(num_pos + 1, hidden_dim)

    def forward(self, feats):
        low_level_feats = torch.cat(feats[0:6], 1)
        medium_level_feats = torch.cat(feats[6:12], 1)
        high_level_feats = torch.cat(feats[12:18], 1)
        batch_size = low_level_feats.shape[0]
        low_level_quality_embedding = self.low_level_flatten_conv2d(low_level_feats).permute(0, 2, 1)
        medium_level_quality_embedding = self.medium_level_flatten_conv2d(medium_level_feats).permute(0, 2, 1)
        high_level_quality_embedding = self.high_level_flatten_conv2d(high_level_feats).permute(0, 2, 1)
        extra_quality_embedding = self.quality_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        quality_embedding = torch.cat((extra_quality_embedding, low_level_quality_embedding,
                                       medium_level_quality_embedding, high_level_quality_embedding), 1)

        position_embedding = self.position_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        return quality_embedding + position_embedding


class Transformer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=4,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src, tgt):
        memory = self.encoder(src.permute(1, 0, 2))
        output = self.decoder(tgt.permute(1, 0, 2), memory)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.multihead_self_attention = nn.MultiheadAttention(d_model, nhead)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.multihead_self_attention(query=src, key=src, value=src)[0]
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + src2)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.multihead_self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        tgt2 = self.multihead_self_attention(query=tgt, key=tgt, value=tgt)[0]
        tgt = self.norm1(tgt + tgt2)

        tgt2 = self.multihead_attention(query=tgt, key=memory, value=memory)[0]
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + tgt2)

        return tgt


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MLPHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp_head(x).squeeze()


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
    def __init__(self, cfg):
        super().__init__()

        if cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'low':
            self.feat_proj = FeatureProjection(num_pos=21 * 21, hidden_dim=cfg.MODEL.TRANSFORMER_DIM)
        elif cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'medium':
            self.feat_proj = FeatureProjection(num_pos=10 * 10, input_dim=6528, hidden_dim=cfg.MODEL.TRANSFORMER_DIM)
        elif cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'high':
            self.feat_proj = FeatureProjection(num_pos=4 * 4, input_dim=12480, hidden_dim=cfg.MODEL.TRANSFORMER_DIM)
        elif cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'mixed':
            self.feat_proj = MixedFeatureProjection(hidden_dim=cfg.MODEL.TRANSFORMER_DIM)
        else:
            self.feat_proj = FeatureProjection(num_pos=21 * 21, hidden_dim=cfg.MODEL.TRANSFORMER_DIM)

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

        feat_dim = {
            'low': 320,
            'medium': 1088,
            'high': 2080,
            'mixed': 2080
        }

        if cfg.MODEL.FEAT_EXTRACTOR_LEVEL == 'mixed':
            self.feat_extractor = MixedFeatureExtractorInceptionResNetV2()
        else:
            self.feat_extractor = FeatureExtractorInceptionResNetV2(level=cfg.MODEL.FEAT_EXTRACTOR_LEVEL)

        self.discriminator = Discriminator(input_dim=feat_dim[cfg.MODEL.FEAT_EXTRACTOR_LEVEL])
        self.classifier = Classifier(input_dim=feat_dim[cfg.MODEL.FEAT_EXTRACTOR_LEVEL])
        self.evaluator = Evaluator(cfg)

    def forward(self, ref_img, dist_img):
        ref_feat = self.feat_extractor(ref_img)
        dist_feat = self.feat_extractor(dist_img)
        return self.discriminator(dist_feat[-1]).view(-1), self.classifier(dist_feat[-1]), self.evaluator(ref_feat,
                                                                                                          dist_feat)
