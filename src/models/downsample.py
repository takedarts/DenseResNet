from .modules import ChannelPad, DropBlock
import torch.nn as nn


class NoneDownsample(nn.Identity):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, ** kwargs):
        super().__init__()


class BasicDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        if stride != 1 or in_channels != out_channels:
            super().__init__(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False),
                normalization(out_channels),
                DropBlock() if dropblock else nn.Identity())
        else:
            super().__init__()


class TweakedDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        modules = []

        if stride != 1:
            modules.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))

        if in_channels != out_channels:
            modules.extend([
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False),
                normalization(out_channels),
                DropBlock() if dropblock else nn.Identity()])

        super().__init__(*modules)


class AverageDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride,
                 normalization, activation, dropblock, **kwargs):
        modules = []

        if stride != 1:
            modules.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))

        if in_channels != out_channels:
            modules.append(ChannelPad(out_channels - in_channels))

        super().__init__(*modules)
