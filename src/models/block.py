#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
from .config import CONFIG
from .modules import DropBlock, ShakeDrop, SignalAugmentation, SEModule, StochasticDepth
from .downsample import NoneDownsample
from .junction import NoneJunction


class Block(nn.Module):

    def __init__(self, pre_operation, post_operation, block_activation,
                 index, length, in_channels, out_channels, stride,
                 operation, downsample, junction, normalization, activation,
                 semodule, shakedrop, stochdepth, sigaug, **kwargs):
        super().__init__()

        modules = [
            pre_operation,
            operation(
                in_channels, out_channels, stride=stride,
                normalization=normalization, activation=activation, **kwargs),
            post_operation]

        if semodule:
            modules.append(SEModule(out_channels, CONFIG.semodule_reduction, activation))

        self.op = nn.Sequential(
            *modules,
            SignalAugmentation(std=sigaug),
            ShakeDrop(drop_prob=shakedrop * (index + 1) / length),
            StochasticDepth(drop_prob=stochdepth * (index + 1) / length))

        self.downsample = downsample(
            in_channels, out_channels, stride=stride,
            normalization=normalization, **kwargs)

        self.junction = junction(
            index + 1, out_channels,
            normalization=normalization, activation=activation, **kwargs)

        self.activation = block_activation

    def forward(self, x):
        # operation
        y = self.op(x[-1])

        # junction
        y, x = self.junction(y, x[:-1] + [self.downsample(x[-1])])

        # output
        x.append(self.activation(y))

        return x


class BasicBlock(Block):

    def __init__(self, index, length, in_channels, out_channels, stride,
                 operation, downsample, junction, normalization, activation, **kwargs):
        super().__init__(
            nn.Identity(),
            nn.Sequential(normalization(out_channels), DropBlock()),
            activation(inplace=True),
            index, length, in_channels, out_channels, stride,
            operation, downsample, junction, normalization, activation, **kwargs)


class PreActBlock(Block):

    def __init__(self, index, length, in_channels, out_channels, stride,
                 operation, downsample, junction, normalization, activation, **kwargs):
        super().__init__(
            nn.Sequential(normalization(in_channels), DropBlock(), activation(inplace=True)),
            nn.Identity(),
            nn.Identity(),
            index, length, in_channels, out_channels, stride,
            operation, downsample, junction, normalization, activation, **kwargs)


class PyramidActBlock(Block):

    def __init__(self, index, length, in_channels, out_channels, stride,
                 operation, downsample, junction, normalization, activation, **kwargs):
        super().__init__(
            normalization(in_channels),
            normalization(out_channels),
            nn.Identity(),
            index, length, in_channels, out_channels, stride,
            operation, downsample, junction, normalization, activation, **kwargs)


class MobileNetBlock(Block):

    def __init__(self, index, length, in_channels, out_channels, stride,
                 operation, downsample, junction, normalization, activation, **kwargs):
        if downsample == NoneDownsample:
            if stride != 1 or in_channels != out_channels:
                junction = NoneJunction

        super().__init__(
            nn.Identity(),
            normalization(out_channels),
            nn.Identity(),
            index, length, in_channels, out_channels, stride,
            operation, downsample, junction, normalization, activation, **kwargs)
