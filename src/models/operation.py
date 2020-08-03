#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .modules import DropBlock, SEModule, HSigmoid
import torch.nn as nn
import math


class BasicOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=3, padding=1,
                stride=1, groups=1, bias=False))


class BottleneckOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=stride, groups=1, bias=False),
            normalization(channels),
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=1, groups=groups, bias=False),
            normalization(channels),
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False))


class TweakedOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck,
                 normalization, activation, **kwargs):
        channels = round(out_channels / bottleneck)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False),
            normalization(channels),
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                stride=stride, groups=groups, bias=False),
            normalization(channels),
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False))


class MobileNetOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel, stride, expansion,
                 normalization, activation, seoperation, **kwargs):
        channels = int(in_channels * expansion)
        modules = []

        if in_channels != channels:
            modules.extend([
                nn.Conv2d(
                    in_channels, channels, kernel_size=1, padding=0,
                    stride=1, groups=1, bias=False),
                normalization(channels),
                DropBlock(),
                activation(inplace=True)])

        modules.extend([
            nn.Conv2d(
                channels, channels, kernel_size=kernel, padding=kernel // 2,
                stride=stride, groups=channels, bias=False),
            normalization(channels)])

        if seoperation:
            modules.append(SEModule(channels, 4, nn.ReLU, lambda: HSigmoid(inplace=True)))

        modules.extend([
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False)])

        super().__init__(*modules)


class SplitAttentionModule(nn.Module):

    def __init__(self, out_channels, radix, groups,
                 normalization, activation, reduction=4):
        super().__init__()
        channels = max(out_channels * radix // reduction, 1)
        channels = math.ceil(channels / 8) * 8

        self.op = nn.Sequential(
            nn.Conv2d(
                out_channels, channels, 1, padding=0, groups=groups, bias=True),
            normalization(channels),
            activation(inplace=True),
            nn.Conv2d(
                channels, out_channels * radix, 1, padding=0, groups=groups, bias=True))

        self.radix = radix

    def forward(self, x):
        w = x.reshape(x.shape[0], self.radix, -1, *x.shape[2:])
        w = w.sum(dim=1).mean(dim=(2, 3), keepdims=True)
        w = self.op(w)
        w = w.reshape(w.shape[0], self.radix, -1, *w.shape[2:])
        w = w.softmax(dim=1)

        x = x.reshape(*w.shape[:3], *x.shape[2:])
        x = (x * w).sum(dim=1)

        return x


class SplitAttentionOperation(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, radix, groups, bottleneck,
                 normalization, activation, **kwargs):
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
            DropBlock(),
            activation(inplace=True),
            nn.Conv2d(
                channels, channels * radix, kernel_size=3, padding=1,
                stride=1, groups=groups * radix, bias=False),
            normalization(channels * radix),
            DropBlock(),
            activation(inplace=True),
            SplitAttentionModule(
                channels, radix=radix, groups=groups,
                normalization=normalization, activation=activation),
            downsample,
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0,
                stride=1, groups=1, bias=False))
