import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn as nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.slices = nn.ModuleList([])

    def forward(self, x):
        feats = []
        for submodule in self.slices:
            x = submodule(x)
            feats.append(x)

        return tuple(feats)


class InceptionResNetV2Backbone(Backbone):
    def __init__(self, level='low'):
        super(InceptionResNetV2Backbone, self).__init__()

        inception_resnet_v2_pretrained = timm.create_model('inception_resnet_v2', pretrained=True)
        inception_resnet_v2_pretrained_features = list(inception_resnet_v2_pretrained.children())

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


class VGG16Backbone(Backbone):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        slice1 = nn.Sequential()
        slice2 = nn.Sequential()
        slice3 = nn.Sequential()
        slice4 = nn.Sequential()
        slice5 = nn.Sequential()

        for x in range(0, 4):
            slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slices.append(slice1)

        slice2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            slice2.add_module(str(x), vgg_pretrained_features[x])
        self.slices.append(slice2)

        slice3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            slice3.add_module(str(x), vgg_pretrained_features[x])
        self.slices.append(slice3)

        slice4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slices.append(slice4)

        slice5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            slice5.add_module(str(x), vgg_pretrained_features[x])
        self.slices.append(slice5)


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        x = x ** 2
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return torch.sqrt(out + 1e-12)
