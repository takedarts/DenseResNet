import torch
import torch.nn as nn
import math


class AFFModule(nn.Module):
    '''Attentional Feature Fusion
    [paper] https://arxiv.org/abs/2009.14082
    [implementation] https://github.com/YimianDai/open-aff
    '''

    def __init__(self, channels, reduction,
                 normalization=nn.BatchNorm2d, activation=nn.ReLU, sigmoid=nn.Sigmoid):
        super().__init__()
        hidden_channels = math.ceil(max(channels // reduction, 1) / 8) * 8

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            normalization(hidden_channels),
            activation(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0, bias=False),
            normalization(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            normalization(hidden_channels),
            activation(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0, bias=False),
            normalization(channels),
        )

    def forward(self, x, y):
        w = x + y
        w = torch.sigmoid(self.local_att(w) + self.global_att(w))

        return 2 * w * x, 2 * (1 - w) * y
