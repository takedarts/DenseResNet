#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max=1, T_up=0, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_up = T_up
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_up:
            return [(self.last_epoch + 1) / (self.T_up + 1) * base_lr
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
