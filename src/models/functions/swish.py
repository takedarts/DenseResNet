import torch
import torch.nn as nn
import torch.autograd as autograd


class SwishFunction(autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)

        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = SwishFunction.apply


def h_swish(x, inplace=False):
    return x * nn.functional.relu6(x + 3, inplace=inplace) / 6
