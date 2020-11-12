import torch.nn as nn
import collections
import math


class SEModule(nn.Module):

    def __init__(self, channels, reduction, activation=nn.ReLU, sigmoid=nn.Sigmoid):
        super().__init__()
        hidden_channels = math.ceil(max(channels // reduction, 1) / 8) * 8

        self.op = nn.Sequential(collections.OrderedDict([
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('conv1', nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0)),
            ('sigmoid', sigmoid()),
        ]))

    def forward(self, x):
        return x * self.op(x)
