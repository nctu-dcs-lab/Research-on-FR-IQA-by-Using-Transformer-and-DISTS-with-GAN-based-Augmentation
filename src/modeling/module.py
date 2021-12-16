import copy

import timm
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        inception_resnet_v2_pretrained = timm.create_model('inception_resnet_v2', pretrained=pretrained)
        inception_resnet_v2_pretrained_features = list(inception_resnet_v2_pretrained.children())

        self.slice1 = nn.Sequential(*inception_resnet_v2_pretrained_features[:8])
        self.slice2 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][:2])
        self.slice3 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][2:4])
        self.slice4 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][4:6])
        self.slice5 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][6:8])
        self.slice6 = nn.Sequential(*inception_resnet_v2_pretrained_features[8][8:10])

    def forward(self, x):
        mixed_5b = self.slice1(x)
        block35_2 = self.slice2(mixed_5b)
        block35_4 = self.slice3(block35_2)
        block35_6 = self.slice4(block35_4)
        block35_8 = self.slice5(block35_6)
        block35_10 = self.slice6(block35_8)

        return torch.cat((mixed_5b, block35_2, block35_4, block35_6, block35_8, block35_10), 1)


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
        batch_size = feat.shape[0]
        quality_embedding = self.flatten_conv2d(feat).permute(0, 2, 1)
        extra_quality_embedding = self.quality_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        quality_embedding = torch.cat((extra_quality_embedding, quality_embedding), 1)

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
    def __init__(self):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifer = nn.Sequential(
            nn.Linear(1920, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.avgpool(feat).view(feat.size(0), -1)
        return self.classifer(feat)


class Evaluator(nn.Module):
    """
    Learned perceptual metric
    """

    def __init__(self):
        super().__init__()

        self.feat_proj = FeatureProjection(num_pos=441, hidden_dim=128)
        self.transformer = Transformer(d_model=128,
                                       nhead=4,
                                       num_encoder_layers=1,
                                       num_decoder_layers=1,
                                       dim_feedforward=1024)
        self.mlp_head = MLPHead(in_dim=128, hidden_dim=128)

    def forward(self, ref_feat, dist_feat):
        diff_feat = ref_feat - dist_feat

        ref_proj_feat = self.feat_proj(ref_feat)
        diff_proj_feat = self.feat_proj(diff_feat)

        return self.mlp_head(self.transformer(diff_proj_feat, ref_proj_feat)[0])


class Classifier(nn.Module):
    def __init__(self, num_classes=116):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifer = nn.Sequential(
            nn.Linear(1920, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, feat):
        feat = self.avgpool(feat).view(feat.size(0), -1)
        return self.classifer(feat)


class MultiTask(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.feature_extractor = FeatureExtractor(pretrained=pretrained)
        self.discriminator = Discriminator()
        self.classifier = Classifier()
        self.evaluator = Evaluator()

    def forward(self, ref_img, dist_img):
        ref_feat = self.feature_extractor(ref_img)
        dist_feat = self.feature_extractor(dist_img)
        return self.discriminator(dist_feat).view(-1), self.classifier(dist_feat), self.evaluator(ref_feat, dist_feat)