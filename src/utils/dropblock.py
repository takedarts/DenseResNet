#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models.modules import DropBlock


class DropBlockController(object):

    @staticmethod
    def update_module(module, prob):
        if isinstance(module, DropBlock):
            module.drop_prob = prob
        else:
            for child in module.children():
                DropBlockController.update_module(child, prob)

    def __init__(self, max_prob, max_epoch, last_epoch=-1):
        self.max_prob = max_prob
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch

    @property
    def prob(self):
        if self.last_epoch + 1 < self.max_epoch - 1:
            epoch = min(max(self.last_epoch + 1, 0), self.max_epoch - 1)
            prob = self.max_prob * epoch / (self.max_epoch - 1)
        else:
            prob = self.max_prob

        return prob

    def update(self, model):
        prob = self.prob

        for block in model.blocks:
            DropBlockController.update_module(block, prob)

    def step(self):
        self.last_epoch += 1

    def state_dict(self):
        return {
            'max_prob': self.max_prob,
            'max_epoch': self.max_epoch,
            'last_epoch': self.last_epoch}

    def load_state_dict(self, state):
        self.max_prob = state['max_prob']
        self.max_epoch = state['max_epoch']
        self.last_epoch = state['last_epoch']
