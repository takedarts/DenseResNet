#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy


class ShakeDropFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x, drop_prob, alpha_range=[-1, 1], training=True):
        if training:
            gate = torch.tensor(int(numpy.random.rand() >= drop_prob), device=x.device)
            ctx.save_for_backward(gate)

            if gate == 0:
                alpha = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)  # @UndefinedVariable
                alpha = alpha.uniform_(*alpha_range).reshape(alpha.shape[0], 1, 1, 1)
                return alpha * x
            else:
                return x

        else:
            return (1 - drop_prob) * x

    @staticmethod
    def backward(ctx, grad):
        gate = ctx.saved_tensors[0]

        if gate == 0:
            beta = torch.empty(grad.shape[0], device=grad.device, dtype=grad.dtype)  # @UndefinedVariable
            beta = beta.uniform_(0, 1).reshape(beta.shape[0], 1, 1, 1)

            return beta * grad, None, None, None
        else:
            return grad, None, None, None


shakedrop = ShakeDropFunction.apply


class ShakeDrop(nn.Module):

    def __init__(self, drop_prob, alpha_range=[-1, 1]):
        super().__init__()
        self.drop_prob = drop_prob
        self.alpha_range = alpha_range

    def forward(self, x):
        if self.drop_prob != 0:
            return shakedrop(x, self.drop_prob, self.alpha_range, self.training)
        else:
            return x

    def extra_repr(self):
        return 'drop_prob={}, alpha_range={}'.format(self.drop_prob, self.alpha_range)
