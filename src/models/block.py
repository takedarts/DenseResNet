import torch.nn as nn
from .config import CONFIG
from .modules import ShakeDrop, SignalAugmentation, SEModule, StochasticDepth
from .downsample import NoneDownsample
from .junction import NoneJunction


class _Block(nn.Module):

    def __init__(self, index, settings, operation, downsample, junction, subsequent,
                 normalization, activation, semodule, dropblock,
                 shakedrop, stochdepth, signalaugment, **kwargs):
        super().__init__()
        in_channels, out_channels, stride = settings[index]

        # downsample
        self.downsample = downsample(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation,
            dropblock=dropblock, **kwargs)

        # convolution layers
        self.operation = operation(
            in_channels, out_channels, stride=stride,
            normalization=normalization, activation=activation,
            dropblock=dropblock, **kwargs)

        # attention modules
        if semodule:
            self.semodule = SEModule(
                out_channels, CONFIG.semodule_reduction, activation=activation)

        # noise
        self.noise = nn.Sequential(
            SignalAugmentation(std=signalaugment),
            ShakeDrop(drop_prob=shakedrop * (index + 1) / len(settings)),
            StochasticDepth(drop_prob=stochdepth * (index + 1) / len(settings)))

        # junction
        self.junction = junction(
            index, settings, normalization=normalization, activation=activation, **kwargs)

        # activation after a block
        self.subsequent = subsequent

    def forward(self, x):
        # operation
        z = self.downsample(x[-1])
        y = self.operation(x[-1])

        # attention
        if hasattr(self, 'semodule'):
            y = self.semodule(y)

        # noise
        y = self.noise(y)

        # junction
        y, x = self.junction(y, x[:-1] + [z])

        # output
        x.append(self.subsequent(y))

        return x


class BasicBlock(_Block):

    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        _, out_channels, _ = settings[index]

        super().__init__(
            index, settings, operation, downsample, junction, nn.ReLU(inplace=True), **kwargs)


class PreActBlock(_Block):

    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        in_channels, _, _ = settings[index]

        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)


class MobileNetBlock(_Block):

    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        in_channels, out_channels, stride = settings[index]

        if downsample == NoneDownsample:
            if stride != 1 or in_channels != out_channels:
                junction = NoneJunction

        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)


class DenseNetBlock(_Block):

    def __init__(self, index, settings, operation, downsample, junction, **kwargs):
        in_channels, _, stride = settings[index]

        if stride != 1:
            junction = NoneJunction
            kwargs['semodule'] = False
            kwargs['shakedrop'] = 0.0
            kwargs['stochdepth'] = 0.0
            kwargs['signalaugment'] = 0.0

        super().__init__(
            index, settings, operation, downsample, junction, nn.Identity(), **kwargs)
