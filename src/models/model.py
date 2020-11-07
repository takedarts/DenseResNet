from .parameter import PARAMETERS
import torch.nn as nn
import itertools

CONV_CLASSES = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
]


def _is_conv_instance(obj):
    for cls in CONV_CLASSES:
        if isinstance(obj, cls):
            return True

    return False


class Model(nn.Module):

    def __init__(self, stem, block, head, classifier, layers,
                 stem_channels, head_channels, dropout, **kwargs):
        super().__init__()

        # make blocks
        channels = [stem_channels] + [c for c, _, _ in layers]
        settings = [(ic, oc, s) for ic, oc, (_, s, _) in zip(channels[:-1], channels[1:], layers)]
        dropblocks = list(itertools.accumulate(s - 1 for _, s, _ in layers))
        dropblocks = [v >= dropblocks[-1] - 1 for v in dropblocks]
        blocks = []

        for i, ((_, _, params), dropblock) in enumerate(zip(layers, dropblocks)):
            block_kwargs = kwargs.copy()
            block_kwargs.update(params)
            blocks.append(block(i, settings, dropblock=dropblock, ** block_kwargs))

        # modules
        self.stem = stem(stem_channels, **kwargs)
        self.blocks = nn.ModuleList(blocks)
        self.head = head(channels[-1], head_channels, **kwargs)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.classifier = classifier(head_channels, **kwargs)

        # initialize weights
        for m in self.modules():
            if _is_conv_instance(m):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_features(self, x):
        x = [self.stem(x)]
        f = []

        for i, block in enumerate(self.blocks):
            x = block(x)
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
        'signalaugment': 0.0}

    model_params.update(PARAMETERS[dataset_name][model_name])
    model_params.update(kwargs)

    return Model(**model_params)
