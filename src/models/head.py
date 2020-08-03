#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn


class BasicHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(nn.Identity())


class PreActHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        super().__init__(
            normalization(in_channels),
            activation(inplace=True))


class MobileNetV2Head(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            normalization(out_channels),
            activation(inplace=True))


class MobileNetV3Head(nn.Sequential):

    def __init__(self, in_channels, out_channels, normalization, activation, **kwargs):
        channels = round(out_channels * 0.75)

        super().__init__(
            nn.Conv2d(
                in_channels, channels, kernel_size=1, padding=0, stride=1, bias=False),
            normalization(channels),
            activation(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True),
            activation(inplace=True))
