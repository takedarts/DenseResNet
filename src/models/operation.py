from .modules import DropBlock, SEModule, SKConv2d, BlurPool2d, SplitAttentionModule
import torch.nn as nn


class BasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class BottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class SelectedKernelOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class PreActBasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            normalization(in_channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False))


class SingleActBasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            normalization(in_channels),
            nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class SingleActBottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(
            normalization(in_channels),
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class TweakedBottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            BlurPool2d(channels, stride=stride) if stride != 1 else nn.Identity(),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class TweakedSlectedKernelOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups),
            DropBlock() if dropblock else nn.Identity(),
            BlurPool2d(channels, stride=stride) if stride != 1 else nn.Identity(),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity())


class MobileNetOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel, stride, expansion,
                 normalization, activation, dropblock, seoperation, sesigmoid, **kwargs):
        channels = int(in_channels * expansion)
        modules = []

        if in_channels != channels:
            modules.extend([
                nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False),
                normalization(channels),
                DropBlock() if dropblock else nn.Identity(),
                activation(inplace=True)])

        modules.extend([
            nn.Conv2d(
                channels, channels, kernel_size=kernel, padding=kernel // 2,
                stride=stride, groups=channels, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity()])

        if seoperation:
            modules.append(SEModule(
                channels, 4, activation=nn.ReLU, sigmoid=sesigmoid))

        modules.extend([
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity()])

        super().__init__(*modules)


class SplitAttentionOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        if stride == 1:
            downsample = nn.Identity()
        else:
            downsample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels * radix, kernel_size=3, padding=1,
                stride=1, groups=groups * radix, bias=False),
            normalization(channels * radix),
            DropBlock() if dropblock else nn.Identity(),
            activation(inplace=True),
            SplitAttentionModule(
                channels, radix=radix, groups=groups,
                normalization=normalization, activation=activation),
            downsample,
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(out_channels),
            DropBlock() if dropblock else nn.Identity())


class DenseNetOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, growth, expansion,
                 normalization, activation, dropblock, **kwargs):
        if stride != 1:
            super().__init__(
                normalization(in_channels),
                activation(inplace=True),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=stride))
        else:
            channels = growth * expansion
            super().__init__(
                normalization(in_channels),
                DropBlock() if dropblock else nn.Identity(),
                activation(inplace=True),
                nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False),
                normalization(channels),
                DropBlock() if dropblock else nn.Identity(),
                activation(inplace=True),
                nn.Conv2d(
                    channels, growth, kernel_size=3, padding=1,
                    stride=1, bias=False))
