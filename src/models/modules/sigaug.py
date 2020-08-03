import torch
import torch.nn as nn
import torch.autograd as autograd


class SignalAugmentFunction(autograd.Function):

    @staticmethod
    def forward(ctx, x, std, dims=1):  # @UnusedVariable
        if std != 0:
            size = list(x.shape[:dims]) + [1] * (len(x.shape) - dims)
            noise = torch.randn(size, device=x.device, requires_grad=False) * std + 1.0  # @UndefinedVariable

            return x * noise
        else:
            return x

    @staticmethod
    def backward(ctx, grad):  # @UnusedVariable
        return grad, None, None


signal_augment = SignalAugmentFunction.apply


class SignalAugmentation(nn.Module):

    def __init__(self, std, dims=1):
        super().__init__()
        self.std = std
        self.dims = dims

    def forward(self, x):
        if self.training and self.std != 0:
            return signal_augment(x, self.std, self.dims)
        else:
            return x

    def extra_repr(self):
        return 'std={}, dim={}'.format(self.std, self.dims)
