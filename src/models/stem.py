import torch.nn as nn
import math


class BasicSmallStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels),
            activation(inplace=True))


class PreActSmallStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels))


class BasicLargeStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(
            nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            normalization(out_channels),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2))


class TweakedLargeStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        mid_channels = max(out_channels // 2, 1)
        mid_channels = math.ceil(mid_channels / 8) * 8

        super().__init__(
            nn.Conv2d(3, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            normalization(mid_channels),
            activation(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            normalization(mid_channels),
            activation(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            normalization(out_channels),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2))


class MobileNetStem(nn.Sequential):

    def __init__(self, out_channels, normalization, activation, **kwargs):
        super().__init__(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            normalization(out_channels),
            activation(inplace=True))
