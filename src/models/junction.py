from .config import CONFIG
from .modules import Reshape
from .functions import adjusted_concat, adjusted_stack

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


class ConcatJunction(nn.Module):

    def __init__(self, num_inputs, in_channels, normalization, activation, **kwargs):
        super().__init__()

    def forward(self, y, x):
        return torch.cat([y, x[-1]], dim=1), x  # @UndefinedVariable


class GatedFunction(autograd.Function):

    @staticmethod
    def forward(ctx, w, *xs):
        ctx.save_for_backward(w, *xs)

        w = w.unsqueeze(3).unsqueeze(4)
        y = adjusted_stack(xs, channels=w.shape[2])

        return (w * y).sum(dim=1, dtype=y.dtype)

    @staticmethod
    def backward(ctx, g):
        w = ctx.saved_variables[0]
        xs = ctx.saved_variables[1:]

        y = adjusted_stack(xs, channels=w.shape[2])
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
        idxs = list(range(max(len(x) - self.connections, 0), len(x)))

        for i in idxs:
            # make a list of inbounds
            if not isinstance(x[i], list):
                x[i] = [x[i], x[i].mean(dim=(2, 3), keepdims=True)]

            # adjust tensor size
            while x[i][0].shape[2:] != y.shape[2:]:
                x[i][0] = nn.functional.avg_pool2d(x[i][0], kernel_size=2)

        # make weights
        w = adjusted_concat([y.mean(dim=(2, 3), keepdims=True)] + [x[i][1] for i in idxs])
        w = self.op(w)
        w1, w2 = w.split([1, int(w.shape[1] - 1)], dim=1)
        w1, w2 = w1.sigmoid(), w2.softmax(dim=1)

        # adapt weights to features
        z = GatedFunction.apply(w2, *[x[i][0] for i in idxs])
        y = y * w1.squeeze(1).unsqueeze(2).unsqueeze(3)

        # save weights
        if CONFIG.save_weights:
            self.weights = w2.detach().cpu().numpy()  # @UndefinedVariable
            self.weights = self.weights.reshape(*self.weights.shape[:2], -1)
            self.indexes = idxs

        # add features
        if z.shape[1] < y.shape[1]:
            y = y + nn.functional.pad(z, (0, 0, 0, 0, 0, int(y.shape[1] - z.shape[1])))
        elif z.shape[1] > y.shape[1]:
            y = y + z[:, :y.shape[1]]
        else:
            y = y + z

        return y, x
