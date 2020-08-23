import torch.nn as nn


class BasicClassifier(nn.Sequential):

    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=0, bias=True))
