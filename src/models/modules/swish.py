from ..functions import swish, h_swish
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return swish(x)


class HSwish(nn.Module):

    def __init__(self, inplace=False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return h_swish(x, inplace=self.inplace)

    def extra_repr(self):
        return 'inplace={}'.format(self.inplace)
