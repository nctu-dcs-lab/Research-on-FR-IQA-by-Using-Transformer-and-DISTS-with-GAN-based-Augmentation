import timm
from torch import nn as nn


class FeatureExtractionInceptionResNetV2(nn.Module):
    def __init__(self, level='low'):
        super().__init__()

        inception_resnet_v2_pretrained = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2_pretrained_features = list(inception_resnet_v2_pretrained.children())

        self.slices = nn.ModuleList([])

        if level == 'low':  # low level feature extraction backbone
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[:8]))
            for i in range(0, 10, 2):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[8][i:i + 2]))

        elif level == 'medium':  # medium level feature extraction backbone
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[:10]))
            for i in range(0, 20, 4):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[10][i:i + 4]))

        elif level == 'high':  # high level feature extraction backbone
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[:12]))
            for i in range(0, 8, 2):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[12][i:i + 2]))
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[12][8:],
                                             inception_resnet_v2_pretrained_features[13]))
        else:  # mixed level feature extraction backbone
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[:8]))
            for i in range(0, 10, 2):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[8][i:i + 2]))

            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[9:10]))
            for i in range(0, 20, 4):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[10][i:i + 4]))

            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[11:12]))
            for i in range(0, 8, 2):
                self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[12][i:i + 2]))
            self.slices.append(nn.Sequential(*inception_resnet_v2_pretrained_features[12][8:],
                                             inception_resnet_v2_pretrained_features[13]))

    def forward(self, x):
        feats = []
        for submodule in self.slices:
            x = submodule(x)
            feats.append(x)

        return tuple(feats)
