#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
from .parameter import PARAMETERS


class Model(nn.Module):

    def __init__(self, stem, block, head, classifier,
                 layers, stem_channels, head_channels,
                 dropout, features, **kwargs):
        super().__init__()

        # make blocks
        blocks = []
        channels = stem_channels

        for i, (out_channels, stride, params) in enumerate(layers):
            block_kwargs = kwargs.copy()
            block_kwargs.update(params)

            blocks.append(
                block(i, len(layers), channels, out_channels, stride, **block_kwargs))
            channels = out_channels

        # modules
        self.stem = stem(stem_channels, **kwargs)
        self.blocks = nn.ModuleList(blocks)
        self.head = head(channels, head_channels, **kwargs)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout != 0 else nn.Identity()
        self.classifier = classifier(head_channels, **kwargs)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # features
        self.feature_indexes = features

    def get_features(self, x):
        x = [self.stem(x)]
        f = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.feature_indexes:
                f.append(x[-1])

        y = self.head(x[-1])

        return y, f

    def get_output(self, x, aggregation=True):
        if aggregation:
            x = nn.functional.adaptive_avg_pool2d(x, 1)
            x = self.dropout(x)
            x = self.classifier(x)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.classifier(x)

        return x

    def forward(self, x, aggregation=True):
        x = self.get_features(x)[0]
        x = self.get_output(x, aggregation=aggregation)

        return x


def create_model(dataset_name, model_name, **kwargs):
    model_params = {
        'normalization': nn.BatchNorm2d,
        'activation': nn.ReLU,
        'semodule': False,
        'dropout': 0.0,
        'shakedrop': 0.0,
        'stochdepth': 0.0,
        'sigaug': 0.0,
        'features': set()}

    model_params.update(PARAMETERS[dataset_name][model_name])
    model_params.update(kwargs)

    return Model(**model_params)
