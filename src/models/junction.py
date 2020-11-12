from .config import CONFIG
from .modules import Reshape

import torch
import torch.nn as nn
import torch.autograd as autograd
import collections
import math


class NoneJunction(nn.Module):

    def __init__(self, index, settings, normalization, activation, **kwargs):
        super().__init__()

    def forward(self, y, x):
        return y, x


class BasicJunction(nn.Module):

    def __init__(self, index, settings, normalization, activation, **kwargs):
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

    def __init__(self, index, settings, normalization, activation, **kwargs):
        super().__init__()

    def forward(self, y, x):
        return torch.cat([y, x[-1]], dim=1), x


class GateFunction(autograd.Function):
    @staticmethod
    def forward(ctx, w, *xs):
        ctx.save_for_backward(w, *xs)

        y = torch.stack(xs, dim=1)[:, :, :w.shape[2]]
        w = w[:, :, :, None, None]

        return (w * y).sum(dim=1, dtype=y.dtype)

    @staticmethod
    def backward(ctx, g):
        w = ctx.saved_variables[0]
        xs = ctx.saved_variables[1:]

        y = torch.stack(xs, dim=1)[:, :, :w.shape[2]]
        w = w[:, :, :, None, None]

        g = g[:, None, :, :, :]

        g_w = (g * y).sum(dim=(3, 4), dtype=g.dtype)
        g_xs = [v.squeeze(1) for v in torch.split(g * w, 1, dim=1)]

        for i, (g_x, x) in enumerate(zip(g_xs, xs)):
            if x.shape[1] > g_x.shape[1]:
                g_xs[i] = nn.functional.pad(g_x, (0, 0, 0, 0, 0, x.shape[1] - g_x.shape[1]))

        return (g_w,) + tuple(g_xs)


class GateJunction(nn.Module):

    def __init__(self, inbounds, index, settings, normalization, activation, **kwargs):
        super().__init__()
        self.indexes = inbounds
        self.channels = settings[index][1]

        for _, oc, s in settings[index + 1:]:
            if s == 1:
                self.channels = max(oc, self.channels)
            else:
                break

        in_channels = settings[index][1] + sum(settings[i][1] for i in self.indexes)
        out_channels = settings[index][1]
        mid_channels = math.ceil(max(out_channels // CONFIG.gate_reduction, 1) / 8) * 8

        self.op = nn.Sequential(collections.OrderedDict(m for m in [
            ('conv1', nn.Conv2d(
                in_channels, mid_channels, kernel_size=1,
                padding=0, bias=False)),
            ('norm1', normalization(mid_channels)),
            ('act1', activation(inplace=True)),
            ('conv2', nn.Conv2d(
                mid_channels, (len(self.indexes) + 1) * out_channels, kernel_size=1,
                padding=0, bias=True)),
            ('reshape', Reshape(len(self.indexes) + 1, out_channels)),
        ] if m[1] is not None))

    def forward(self, y, x):
        for i in self.indexes:
            # make a list of inbounds
            if not isinstance(x[i], list):
                feat = x[i].mean(dim=(2, 3), keepdims=True)
                mask = torch.zeros(feat.shape[:2], dtype=feat.dtype, device=feat.device)
                x[i] = [x[i], feat, mask]

            # adjust tensor size
            while x[i][0].shape[2:] != y.shape[2:]:
                x[i][0] = nn.functional.avg_pool2d(x[i][0], kernel_size=2, ceil_mode=True)

            if x[i][0].shape[1] < self.channels:
                padding = self.channels - x[i][0].shape[1]
                x[i][0] = nn.functional.pad(x[i][0], (0, 0, 0, 0, 0, padding))
                x[i][2] = nn.functional.pad(x[i][2], (0, padding), value=1)

        # make weights
        f = y.mean(dim=(2, 3), keepdims=True)
        f = torch.cat([f] + [x[i][1] for i in self.indexes], dim=1)
        m = torch.stack([x[i][2] for i in self.indexes], dim=1)

        w1, w2 = self.op(f).split([1, m.shape[1]], dim=1)
        w1 = w1.sigmoid()
        w2 = (w2 - (m[:, :, :w2.shape[2]] * 1e+8)).softmax(dim=1)

        # adapt weights to features
        z = GateFunction.apply(w2, *[x[i][0] for i in self.indexes])
        y = y * w1.squeeze(1).unsqueeze(2).unsqueeze(3)

        # save weights
        if CONFIG.save_weights:
            self.weights = w2.detach().cpu().numpy()
            self.weights = self.weights.reshape(*self.weights.shape[:2], -1)

        return y + z, x


class DenseJunction(GateJunction):

    def __init__(self, index, settings, normalization, activation, **kwargs):
        connections = min(CONFIG.dense_connections, index + 1)
        super().__init__(
            [index - i for i in range(connections)],
            index, settings, normalization, activation, **kwargs)


class SkipJunction(GateJunction):

    def __init__(self, index, settings, normalization, activation, **kwargs):
        connections = min(CONFIG.skip_connections, int(math.log2(index + 1)) + 1)
        super().__init__(
            [index - (2 ** i) + 1 for i in range(connections)],
            index, settings, normalization, activation, **kwargs)
