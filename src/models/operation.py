from .config import CONFIG
from .modules import DropBlock, SEModule, SKConv2d, BlurPool2d, SplitAttentionModule

import torch.nn as nn
import collections


class BasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(out_channels)),
            ('drop2', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class BottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class SelectedKernelOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class PreActBasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
        ] if m[1] is not None))


class SingleActBasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class SingleActBottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck * groups)

        super().__init__(collections.OrderedDict(m for m in [
            ('norm1', normalization(in_channels)),
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False)),
            ('norm3', normalization(channels)),
            ('drop3', None if not dropblock else DropBlock()),
            ('act3', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm4', normalization(out_channels)),
            ('drop4', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class TweakedBottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('pool', None if stride == 1 else BlurPool2d(channels, stride=stride)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class TweakedSlectedKernelOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', SKConv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, radix=radix, groups=groups)),
            ('drop2', None if not dropblock else DropBlock()),
            ('pool', None if stride == 1 else BlurPool2d(channels, stride=stride)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class MobileNetOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel, stride, expansion,
                 normalization, activation, dropblock, seoperation, sesigmoid, **kwargs):
        channels = int(in_channels * expansion)
        modules = []

        if in_channels != channels:
            modules.extend([
                ('conv1', nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('norm1', normalization(channels)),
                ('drop1', None if not dropblock else DropBlock()),
                ('act1', activation(inplace=True)),
            ])

        modules.extend([
            ('conv2', nn.Conv2d(
                channels, channels, kernel_size=kernel, padding=kernel // 2,
                stride=stride, groups=channels, bias=False)),
            ('norm2', normalization(channels)),
            ('drop2', None if not dropblock else DropBlock()),
            ('semodule', None if not seoperation else SEModule(
                channels, reduction=CONFIG.semodule_reduction,
                activation=nn.ReLU, sigmoid=sesigmoid)),
            ('act2', activation(inplace=True)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ])

        super().__init__(collections.OrderedDict(m for m in modules if m[1] is not None))


class SplitAttentionOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, dropblock, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm1', normalization(channels)),
            ('drop1', None if not dropblock else DropBlock()),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                channels, channels * radix, kernel_size=3, padding=1,
                stride=1, groups=groups * radix, bias=False)),
            ('norm2', normalization(channels * radix)),
            ('drop2', None if not dropblock else DropBlock()),
            ('act2', activation(inplace=True)),
            ('attention', SplitAttentionModule(
                channels, radix=radix, groups=groups,
                normalization=normalization, activation=activation)),
            ('downsample', None if stride == 1 else nn.AvgPool2d(
                kernel_size=3, stride=stride, padding=1)),
            ('conv3', nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)),
            ('norm3', normalization(out_channels)),
            ('drop3', None if not dropblock else DropBlock()),
        ] if m[1] is not None))


class DenseNetOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, growth, expansion,
                 normalization, activation, dropblock, **kwargs):
        if stride != 1:
            super().__init__(collections.OrderedDict(m for m in [
                ('norm1', normalization(in_channels)),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('pool1', nn.AvgPool2d(kernel_size=2, stride=stride)),
            ] if m[1] is not None))
        else:
            channels = growth * expansion
            super().__init__(collections.OrderedDict(m for m in [
                ('norm1', normalization(in_channels)),
                ('drop1', None if not dropblock else DropBlock()),
                ('act1', activation(inplace=True)),
                ('conv1', nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False)),
                ('norm2', normalization(channels)),
                ('drop2', None if not dropblock else DropBlock()),
                ('act2', activation(inplace=True)),
                ('conv2', nn.Conv2d(
                    channels, growth, kernel_size=3, padding=1,
                    stride=1, bias=False)),
            ] if m[1] is not None))
