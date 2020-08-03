#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import math


class SEModule(nn.Module):

    def __init__(self, channels, reduction, activation, sigmoid=nn.Sigmoid):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        hidden_channels = math.ceil(hidden_channels / 8) * 8

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0),
            activation(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0),
            sigmoid())

    def forward(self, x):
        return x * self.op(x)
