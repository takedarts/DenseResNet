import torch.nn as nn
import torch.optim as optim

NORM_CLASSES = set([
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
])

CONV_CLASSES = set([
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
])


def _isinstance(obj, classes):
    for cls in classes:
        if isinstance(obj, cls):
            return True

    return False


def create_optimizer(model, lr, weight_decay, bias_decay=False):
    if bias_decay:
        return optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    norm_names = set()
    conv_names = set()
    nodecay_params = []
    decay_params = []

    for name, module in model.named_modules():
        if _isinstance(module, NORM_CLASSES):
            norm_names.add(name)
        elif _isinstance(module, CONV_CLASSES):
            conv_names.add(f'{name}.bias')

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif name.rsplit('.', maxsplit=1)[0] in norm_names or name in conv_names:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    parameters = [
        {'params': nodecay_params, 'weight_decay': 0.0},
        {'params': decay_params, 'weight_decay': weight_decay}]

    return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
