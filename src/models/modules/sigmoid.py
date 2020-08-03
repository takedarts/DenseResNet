#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn


def h_sigmoid(x, inplace=False):
    return nn.functional.relu6(x + 3, inplace=inplace) / 6


class HSigmoid(nn.Module):

    def __init__(self, inplace=False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return h_sigmoid(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
