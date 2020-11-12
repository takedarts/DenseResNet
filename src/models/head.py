import torch.nn as nn
import collections


class BasicHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()


class PreActHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('norm', normalization(in_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class MobileNetV2Head(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        super().__init__(collections.OrderedDict(m for m in [
            ('conv', nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)),
            ('norm', normalization(out_channels)),
            ('act', activation(inplace=True)),
        ] if m[1] is not None))


class MobileNetV3Head(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        channels = round(out_channels * 0.75)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0, stride=1, bias=False)),
            ('norm1', normalization(channels)),
            ('act1', activation(inplace=True)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True)),
            ('act2', activation(inplace=True)),
        ] if m[1] is not None))
