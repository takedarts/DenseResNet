#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Config(object):

    def __init__(self):
        self.semodule_reduction = 16
        self.gate_reduction = 8
        self.gate_connections = 4
        self.dropblock_size = 7
        self.save_weights = False


CONFIG = Config()
