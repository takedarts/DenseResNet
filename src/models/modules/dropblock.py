#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.config import CONFIG


class DropBlock(nn.Module):

    def __init__(self, drop_prob=0, block_size=CONFIG.dropblock_size):
        super().__init__()

        assert block_size % 2 == 1

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)  # @UndefinedVariable
        mask = (mask < gamma).float()
        mask = nn.functional.max_pool2d(
            mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1 - mask
        mask *= mask.numel() / mask.sum()

        return x * mask

    def extra_repr(self):
        return 'drop_prob={}, block_size={}'.format(self.drop_prob, self.block_size)
