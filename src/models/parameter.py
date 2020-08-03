#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .stem import BasicSmallStem, PreActSmallStem, BasicLargeStem, TweakedLargeStem
from .stem import MobileNetStem
from .head import BasicHead, PreActHead, MobileNetV2Head, MobileNetV3Head
from .classifier import BasicClassifier
from .block import BasicBlock, PreActBlock, PyramidActBlock, MobileNetBlock
from .operation import BasicOperation, BottleneckOperation, TweakedOperation
from .operation import MobileNetOperation, SplitAttentionOperation
from .downsample import BasicDownsample, TweakedDownsample, AverageDownsample, NoneDownsample
from .junction import BasicJunction, GatedJunction
from .modules import Swish, HSwish

import torch.nn as nn
import itertools
import math


def make_resnet_layers(depths, channels, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    strides = [1] + [2] * (len(depths) - 1)
    layers = []

    for depth, stride in zip(depths, strides):
        layers.append((round(channels * bottleneck), stride, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_resnest_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    strides = [1] + [2] * (len(depths) - 1)
    layers = []

    for depth, stride in zip(depths, strides):
        layers.append((round(channels * bottleneck), stride, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_pyramid_layers(depths, base, alpha, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    depths = list(itertools.accumulate(depths))
    layers = []

    for i in range(depths[-1]):
        channels = round(base + alpha * (i + 1) / depths[-1])
        stride = 2 if i in depths[:-1] else 1
        layers.append((round(channels * bottleneck), stride, params))

    return layers


def make_mobilenet_layer(kernel, channels, stride, expansion,
                         seoperation, activation, divisor=8):
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)

    if new_channels < 0.9 * channels:
        new_channels += divisor

    params = {
        'kernel': kernel,
        'expansion': expansion,
        'seoperation': seoperation,
        'activation': activation}

    return (new_channels, stride, params)


def make_mobilenet_layers(settings, width, depth):
    layers = []

    for kernel, channels, stride, expansion, seoperation, activation, repeats in settings:
        repeats = math.ceil(repeats * depth)
        params = {
            'kernel': kernel,
            'channels': channels * width,
            'expansion': expansion,
            'seoperation': seoperation,
            'activation': activation}
        layers.append(make_mobilenet_layer(stride=stride, **params))
        layers.extend(make_mobilenet_layer(stride=1, **params) for _ in range(repeats - 1))

    return layers


def make_mobilenetv2_layers(width):
    settings = [  # kernel, channels, stride, expansion, SE, activation, repeats
        [3, 16, 1, 1, False, nn.ReLU6, 1],
        [3, 24, 2, 6, False, nn.ReLU6, 2],
        [3, 32, 2, 6, False, nn.ReLU6, 3],
        [3, 64, 2, 6, False, nn.ReLU6, 4],
        [3, 96, 1, 6, False, nn.ReLU6, 3],
        [3, 160, 2, 6, False, nn.ReLU6, 3],
        [3, 320, 1, 6, False, nn.ReLU6, 1]]

    return make_mobilenet_layers(settings, width, 1.0)


def make_mobilenetv3_large_layers(width):
    settings = [  # kernel, channels, stride, expansion, SE, activation, repeats
        [3, 16, 1, 1, False, nn.ReLU, 1],
        [3, 24, 2, 4, False, nn.ReLU, 1],
        [3, 24, 1, 3, False, nn.ReLU, 1],
        [5, 40, 2, 3, True, nn.ReLU, 3],
        [3, 80, 2, 6, False, HSwish, 1],
        [3, 80, 1, 2.5, False, HSwish, 1],
        [3, 80, 1, 2.3, False, HSwish, 2],
        [3, 112, 1, 6, True, HSwish, 2],
        [5, 160, 2, 6, True, HSwish, 3]]

    return make_mobilenet_layers(settings, width, 1.0)


def make_efficientnet_layers(width, depth):
    settings = [  # kernel, channels, stride, expansion, se-module, activation, repeats
        [3, 16, 1, 1, True, Swish, 1],
        [3, 24, 2, 6, True, Swish, 2],
        [5, 40, 2, 6, True, Swish, 2],
        [3, 80, 2, 6, True, Swish, 3],
        [5, 112, 1, 6, True, Swish, 3],
        [5, 192, 2, 6, True, Swish, 4],
        [3, 320, 1, 6, True, Swish, 1]]

    return make_mobilenet_layers(settings, width, depth)


def update_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def update_models(models, **kwargs):
    new_models = {}

    for name, params in models.items():
        new_models[name] = update_params(params, **kwargs)

    return new_models


def resnetd_models(models):
    new_models = {}

    for name, params in models.items():
        if not name.startswith('resnet-'):
            continue

        new_params = params.copy()

        if new_params['stem'] == BasicLargeStem:
            new_params['stem'] = TweakedLargeStem

        if new_params['operation'] == BottleneckOperation:
            new_params['operation'] = TweakedOperation

        if new_params['downsample'] == BasicDownsample:
            new_params['downsample'] = TweakedDownsample

        if new_params == params:
            continue

        new_models[f'resnetd-{name[7:]}'] = new_params

    return new_models


def dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = GatedJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'dense-{name}'] = new_params

    return new_models


large_basic_params = {
    'stem': BasicLargeStem,
    'head': BasicHead,
    'classifier': BasicClassifier,
    'block': BasicBlock,
    'operation': BasicOperation,
    'downsample': BasicDownsample,
    'junction': BasicJunction}

large_models = {
    'resnet-18': update_params(
        large_basic_params,
        layers=make_resnet_layers([2, 2, 2, 2], 64, 1, 1),
        stem_channels=64, head_channels=512),

    'resnet-34': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512),

    'resnet-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation),

    'mobilenetv2-1.0': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6),

    'mobilenetv2-0.5': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(0.5),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6),

    'mobilenetv3-large': update_params(
        large_basic_params,
        layers=make_mobilenetv3_large_layers(1.0),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV3Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=HSwish),

    'efficientnet-b0': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish),

    'efficientnet-b1': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.1),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish),

    'efficientnet-b2': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.1, 1.2),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish),

    'resnest-50': update_params(  # ResNeSt-50-2s1x64d
        large_basic_params,
        layers=make_resnest_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=SplitAttentionOperation),
}

small_basic_params = update_params(large_basic_params, stem=BasicSmallStem)

small_models = {
    'resnet-20': update_params(
        small_basic_params,
        layers=make_resnet_layers([3, 3, 3], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'resnet-110': update_params(
        small_basic_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'wideresnet-28k10': update_params(
        small_basic_params,
        layers=make_resnet_layers([4, 4, 4], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock),

    'wideresnet-40k4': update_params(
        small_basic_params,
        layers=make_resnet_layers([6, 6, 6], 64, 1, 1),
        stem_channels=16, head_channels=256,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock),

    'pyramidnet-110a48': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 48, 1, 1),
        stem_channels=16, head_channels=64,
        stem=PreActSmallStem, head=PreActHead,
        block=PyramidActBlock, downsample=AverageDownsample),

    'pyramidnet-110a270': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 270, 1, 1),
        stem_channels=16, head_channels=286,
        stem=PreActSmallStem, head=PreActHead,
        block=PyramidActBlock, downsample=AverageDownsample),

    'pyramidnet-200a240': update_params(
        small_basic_params,
        layers=make_pyramid_layers([22, 22, 22], 16, 240, 1, 4),
        stem_channels=16, head_channels=1024,
        stem=PreActSmallStem, head=PreActHead,
        block=PyramidActBlock, downsample=AverageDownsample,
        operation=BottleneckOperation),
}

size256_models = {}
size256_models.update(large_models)
size256_models.update(resnetd_models(size256_models))
size256_models.update(dense_models(size256_models))

size64_models = {}
size64_models.update(update_models(
    {n: m for n, m in large_models.items() if m['stem'] == BasicLargeStem},
    stem=BasicSmallStem))
size64_models.update(resnetd_models(size64_models))
size64_models.update(dense_models(size64_models))

size32_models = {}
size32_models.update(small_models)
size32_models.update(resnetd_models(size32_models))
size32_models.update(dense_models(size32_models))

PARAMETERS = {
    'imagenet': update_models(size256_models, num_classes=1000),
    'tinyimagenet': update_models(size64_models, num_classes=200),
    'cifar100': update_models(size32_models, num_classes=100),
    'cifar10': update_models(size32_models, num_classes=10),
}
