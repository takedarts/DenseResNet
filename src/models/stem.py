import torch.nn as nn
import collections
import math


class BasicSmallStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class PreActSmallStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
        ] if m[1] is not None))


class BasicLargeStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, padding=1, stride=2)),
        ] if m[1] is not None))


class TweakedLargeStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        mid_channels = max(out_channels // 2, 1)
        mid_channels = math.ceil(mid_channels / 8) * 8

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(3, mid_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm1', normalization(mid_channels)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            ('norm2', normalization(mid_channels)),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('act3', activation(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, padding=1, stride=2)),
        ] if m[1] is not None))


class MobileNetStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))
