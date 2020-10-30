from .reshape import Reshape
import torch.nn as nn
import math


class SKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, radix=2, groups=1, reduction=2,
                 normalization=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()
        attn_channels = math.ceil(max(out_channels // reduction, 1) / 8) * 8

        self.op = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels * radix, kernel_size=3,
                stride=stride, padding=padding, groups=groups, bias=False),
            normalization(out_channels * radix),
            activation(inplace=True))

        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, attn_channels, kernel_size=1, padding=0, bias=False),
            normalization(attn_channels),
            activation(inplace=True),
            nn.Conv2d(attn_channels, out_channels * radix, kernel_size=1, padding=0, bias=False),
            Reshape(radix, out_channels, 1, 1),
            nn.Softmax(dim=1))

    def forward(self, x):
        y = self.op(x)
        y = y.reshape(y.shape[0], y.shape[1] // x.shape[1], x.shape[1], *y.shape[2:])
        w = self.attn(y.sum(dim=1).mean(dim=(2, 3), keepdims=True))

        return (w * y).sum(dim=1, dtype=y.dtype)
