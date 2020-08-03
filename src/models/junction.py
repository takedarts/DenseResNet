#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .config import CONFIG
from .modules import Reshape

import torch
import torch.nn as nn
import torch.autograd as autograd

import math


class NoneJunction(nn.Module):

    def __init__(self, num_inputs, in_channels, normalization, activation, **kwargs):
        super().__init__()

    def forward(self, y, x):
        return y, x


class BasicJunction(nn.Module):

    def __init__(self, num_inputs, in_channels, normalization, activation, **kwargs):
        super().__init__()

    def forward(self, y, x):
        z = x[-1]

        if z.shape[1] < y.shape[1]:
            y = y + nn.functional.pad(z, (0, 0, 0, 0, 0, int(y.shape[1] - z.shape[1])))
        elif z.shape[1] > y.shape[1]:
            y = y + z[:, :y.shape[1]]
        else:
            y = y + z

        return y, x


class GatedConcatenationFunction(autograd.Function):

    @staticmethod
    def forward(ctx, *xs):
        ctx.channels = [x.shape[1] for x in xs]

        base = xs[0]
        y = torch.zeros(  # @UndefinedVariable
            base.shape[0], base.shape[1] * len(xs), 1, 1,
            device=base.device, dtype=base.dtype)

        for i, x in enumerate(xs):
            o = i * base.shape[1]

            if x.shape[1] > base.shape[1]:
                y[:, o:o + base.shape[1]] = x[:, :base.shape[1]]
            else:
                y[:, o:o + x.shape[1]] = x

        return y

    @staticmethod
    def backward(ctx, g):
        channels = ctx.channels[0]
        g_xs = []

        for i, c in enumerate(ctx.channels):
            o = i * channels
            g_x = g[:, o:o + min(c, channels)]

            if c > channels:
                g_x = nn.functional.pad(g_x, (0, 0, 0, 0, 0, c - channels))

            g_xs.append(g_x)

        return tuple(g_xs)


class GatedAmplificationFunction(autograd.Function):

    @staticmethod
    def _concat(channels, *xs):
        base = xs[-1]
        y = torch.zeros(  # @UndefinedVariable
            base.shape[0], len(xs), channels, *base.shape[2:],
            device=base.device, dtype=base.dtype)

        for i, x in enumerate(xs):
            if x.shape[1] < y.shape[2]:
                y[:, i, :x.shape[1]] = x
            elif x.shape[1] > y.shape[2]:
                y[:, i] = x[:, :y.shape[2]]
            else:
                y[:, i] = x

        return y

    @staticmethod
    def forward(ctx, w, *xs):
        ctx.save_for_backward(w, *xs)

        w = w.unsqueeze(3).unsqueeze(4)
        y = GatedAmplificationFunction._concat(w.shape[2], *xs)

        return (w * y).sum(dim=1, dtype=y.dtype)

    @staticmethod
    def backward(ctx, g):
        w = ctx.saved_variables[0]
        xs = ctx.saved_variables[1:]

        y = GatedAmplificationFunction._concat(w.shape[2], *xs)
        w = w.unsqueeze(3).unsqueeze(4)
        g = g.unsqueeze(1)

        g_w = (g * y).sum(dim=(3, 4), dtype=g.dtype)
        g_xs = [v.squeeze(1) for v in torch.split(g * w, 1, dim=1)]

        for i, (g_x, x) in enumerate(zip(g_xs, xs)):
            if x.shape[1] < g_x.shape[1]:
                g_xs[i] = g_x[:, :x.shape[1]]
            elif x.shape[1] > g_x.shape[1]:
                g_xs[i] = nn.functional.pad(g_x, (0, 0, 0, 0, 0, x.shape[1] - g_x.shape[1]))

        return (g_w,) + tuple(g_xs)


class GatedJunction(nn.Module):

    def __init__(self, num_inputs, in_channels, normalization, activation, **kwargs):
        super().__init__()
        self.connections = min(num_inputs, CONFIG.gate_connections)
        channels = max(in_channels // CONFIG.gate_reduction, 1)
        channels = math.ceil(channels / 8) * 8

        self.op = nn.Sequential(
            nn.Conv2d(
                in_channels * (self.connections + 1), channels, kernel_size=1,
                padding=0, bias=False),
            normalization(channels),
            activation(inplace=True),
            nn.Conv2d(
                channels, in_channels * (self.connections + 1), kernel_size=1,
                padding=0, bias=True),
            Reshape(self.connections + 1, in_channels))

    def forward(self, y, x):
        for i, v in enumerate(x):
            if not isinstance(v, list):
                x[i] = [v, v.mean(dim=(2, 3), keepdims=True)]

        for v in x[-self.connections:]:
            if v[0].shape[2:] != x[-1][0].shape[2:]:
                v[0] = nn.functional.avg_pool2d(v[0], kernel_size=2)

        w = GatedConcatenationFunction.apply(
            y.mean(dim=(2, 3), keepdims=True), *[v[1] for v in x[-self.connections:]])

        w = self.op(w)
        w1, w2 = w.split([1, int(w.shape[1] - 1)], dim=1)
        w1, w2 = w1.sigmoid(), w2.softmax(dim=1)

        z = GatedAmplificationFunction.apply(w2, *[v[0] for v in x[-self.connections:]])
        y = y * w1.squeeze(1).unsqueeze(2).unsqueeze(3)

        if CONFIG.save_weights:
            self.weights = w2.detach().cpu().numpy()  # @UndefinedVariable
            self.weights = self.weights.reshape(*self.weights.shape[:2], -1)

        if z.shape[1] < y.shape[1]:
            y = y + nn.functional.pad(z, (0, 0, 0, 0, 0, int(y.shape[1] - z.shape[1])))
        elif z.shape[1] > y.shape[1]:
            y = y + z[:, :y.shape[1]]
        else:
            y = y + z

        return y, x
