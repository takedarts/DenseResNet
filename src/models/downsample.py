#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn


class NoneDownsample(nn.Identity):

    def __init__(self, in_channels, out_channels, stride, normalization, **kwargs):
        super().__init__()
        self.out_channels = in_channels


class BasicDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, normalization, **kwargs):
        if stride != 1 or in_channels != out_channels:
            super().__init__(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False),
                normalization(out_channels))
        else:
            super().__init__()

        self.out_channels = out_channels


class TweakedDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, normalization, **kwargs):
        modules = []

        if stride != 1:
            modules.append(nn.AvgPool2d(kernel_size=2, stride=stride))

        if in_channels != out_channels:
            modules.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=1, padding=0, bias=False))
            modules.append(normalization(out_channels))

        super().__init__(*modules)
        self.out_channels = out_channels


class AverageDownsample(nn.Sequential):

    def __init__(self, in_channels, out_channels, stride, normalization, **kwargs):
        if stride != 1:
            super().__init__(nn.AvgPool2d(kernel_size=2, stride=stride))
        else:
            super().__init__()

        self.out_channels = in_channels
