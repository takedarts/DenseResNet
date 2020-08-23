import torch.nn as nn
import torch.optim as optim

NORM_CLASSES = set([
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
])


def create_optimizer(model, lr, weight_decay, norm_decay):
    if norm_decay:
        return optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    norm_names = set()
    norm_params = []
    other_params = []

    for name, module in model.named_modules():
        if type(module) in NORM_CLASSES:
            norm_names.add(name)

    for name, param in model.named_parameters():
        if name.rsplit('.', maxsplit=1)[0] in norm_names:
            norm_params.append(param)
        else:
            other_params.append(param)

    parameters = [
        {'params': norm_params, 'weight_decay': 0.0},
        {'params': other_params, 'weight_decay': weight_decay}]

    return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
