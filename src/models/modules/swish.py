#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd


class SwishFunction(autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)  # @UndefinedVariable
        ctx.save_for_backward(i)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)  # @UndefinedVariable

        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = SwishFunction.apply


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return swish(x)


def h_swish(x, inplace=False):
    return x * nn.functional.relu6(x + 3, inplace=inplace) / 6


class HSwish(nn.Module):

    def __init__(self, inplace=False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return h_swish(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
