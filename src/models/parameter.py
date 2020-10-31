from .stem import BasicSmallStem, PreActSmallStem, BasicLargeStem, TweakedLargeStem
from .stem import MobileNetStem
from .head import BasicHead, PreActHead, MobileNetV2Head, MobileNetV3Head
from .classifier import BasicClassifier
from .block import BasicBlock, PreActBlock, MobileNetBlock, DenseNetBlock
from .operation import BasicOperation, BottleneckOperation, SelectedKernelOperation
from .operation import SingleActBasicOperation, SingleActBottleneckOperation
from .operation import TweakedBottleneckOperation, TweakedSlectedKernelOperation
from .operation import MobileNetOperation, SplitAttentionOperation, DenseNetOperation
from .downsample import BasicDownsample, TweakedDownsample, AverageDownsample, NoneDownsample
from .junction import BasicJunction, ConcatJunction, DenseJunction, SkipJunction
from .modules import Swish, HSwish, HSigmoid

import torch.nn as nn
import itertools
import math


def make_resnet_layers(depths, channels, groups, bottleneck):
    params = {'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_resnest_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
        layers.extend((round(channels * bottleneck), 1, params) for _ in range(depth - 1))
        channels *= 2

    return layers


def make_skresnet_layers(depths, channels, radix, groups, bottleneck):
    params = {'radix': radix, 'groups': groups, 'bottleneck': bottleneck}
    layers = []

    for i, depth in enumerate(depths):
        layers.append((round(channels * bottleneck), 1 if i == 0 else 2, params))
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


def make_densenet_layers(depths, channels, growth, expansion):
    params = {'growth': growth, 'expansion': expansion}
    layers = []

    for i, depth, in enumerate(depths):
        if i != 0:
            channels //= 2
            layers.append((channels, 2, params))

        for _ in range(depth):
            channels += growth
            layers.append((channels, 1, params))

    return layers


def update_params(params, **kwargs):
    new_params = params.copy()
    new_params.update(kwargs)

    return new_params


def update_models(models, **kwargs):
    new_models = {}

    for name, params in models.items():
        new_models[name] = update_params(params, **kwargs)

    return new_models


def dense_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = DenseJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Dense-{name}'] = new_params

    return new_models


def skip_models(models):
    new_models = {}

    for name, params in models.items():
        new_params = params.copy()

        if new_params['junction'] == BasicJunction:
            new_params['junction'] = SkipJunction
        else:
            continue

        if new_params['downsample'] == NoneDownsample:
            new_params['downsample'] = AverageDownsample

        new_models[f'Skip-{name}'] = new_params

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
    'ResNet-18': update_params(
        large_basic_params,
        layers=make_resnet_layers([2, 2, 2, 2], 64, 1, 1),
        stem_channels=64, head_channels=512),

    'ResNet-34': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512),

    'ResNet-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation),

    'ResNet-101': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 23, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation),

    'SE-ResNet-34': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 1),
        stem_channels=64, head_channels=512, semodule=True),

    'SE-ResNet-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation, semodule=True),

    'SK-ResNet-50': update_params(
        large_basic_params,
        layers=make_skresnet_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        operation=SelectedKernelOperation),

    'ResNetD-50': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 64, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=TweakedBottleneckOperation),

    'SK-ResNetD-50': update_params(
        large_basic_params,
        layers=make_skresnet_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=TweakedSlectedKernelOperation),

    'ResNeXt-50-32x4d': update_params(
        large_basic_params,
        layers=make_resnet_layers([3, 4, 6, 3], 4, 32, 64),
        stem_channels=64, head_channels=2048,
        operation=BottleneckOperation),

    'ResNeSt-50-2s1x64d': update_params(
        large_basic_params,
        layers=make_resnest_layers([3, 4, 6, 3], 64, 2, 1, 4),
        stem_channels=64, head_channels=2048,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=SplitAttentionOperation),

    'MobileNetV2-1.0': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6,
        seoperation=False, sesigmoid=None),

    'MobileNetV2-0.5': update_params(
        large_basic_params,
        layers=make_mobilenetv2_layers(0.5),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=nn.ReLU6,
        seoperation=False, sesigmoid=None),

    'MobileNetV3-large': update_params(
        large_basic_params,
        layers=make_mobilenetv3_large_layers(1.0),
        stem_channels=16, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV3Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=HSwish,
        seoperation=True, sesigmoid=lambda: HSigmoid(inplace=True)),

    'EfficientNet-B0': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.0),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish,
        seoperation=True, sesigmoid=nn.Sigmoid),

    'EfficientNet-B1': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.0, 1.1),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish,
        seoperation=True, sesigmoid=nn.Sigmoid),

    'EfficientNet-B2': update_params(
        large_basic_params,
        layers=make_efficientnet_layers(1.1, 1.2),
        stem_channels=32, head_channels=1280,
        stem=MobileNetStem, head=MobileNetV2Head,
        block=MobileNetBlock, operation=MobileNetOperation,
        downsample=NoneDownsample, activation=Swish,
        seoperation=True, sesigmoid=nn.Sigmoid),

    'DenseNet-121': update_params(
        large_basic_params,
        layers=make_densenet_layers([6, 12, 24, 16], 64, 32, 4),
        stem_channels=64, head_channels=1024,
        head=PreActHead, block=DenseNetBlock, operation=DenseNetOperation,
        downsample=NoneDownsample, junction=ConcatJunction),

    'DenseNet-169': update_params(
        large_basic_params,
        layers=make_densenet_layers([6, 12, 32, 32], 64, 32, 4),
        stem_channels=64, head_channels=1664,
        head=PreActHead, block=DenseNetBlock, operation=DenseNetOperation,
        downsample=NoneDownsample, junction=ConcatJunction),
}

small_basic_params = update_params(large_basic_params, stem=BasicSmallStem)

small_models = {
    'ResNet-20': update_params(
        small_basic_params,
        layers=make_resnet_layers([3, 3, 3], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'ResNet-110': update_params(
        small_basic_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64),

    'SE-ResNet-110': update_params(
        small_basic_params,
        layers=make_resnet_layers([18, 18, 18], 16, 1, 1),
        stem_channels=16, head_channels=64, semodule=True),

    'WideResNet-28-k10': update_params(
        small_basic_params,
        layers=make_resnet_layers([4, 4, 4], 160, 1, 1),
        stem_channels=16, head_channels=640,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock,
        operation=None),

    'WideResNet-40-k4': update_params(
        small_basic_params,
        layers=make_resnet_layers([6, 6, 6], 64, 1, 1),
        stem_channels=16, head_channels=256,
        stem=PreActSmallStem, head=PreActHead, block=PreActBlock,
        operation=None),

    'ResNeXt-29-8x64d': update_params(
        small_basic_params,
        layers=make_resnet_layers([3, 3, 3], 64, 8, 4),
        stem_channels=64, head_channels=1024,
        operation=BottleneckOperation),

    'ResNeXt-47-32x4d': update_params(
        small_basic_params,
        layers=make_resnet_layers([5, 5, 5], 4, 32, 64),
        stem_channels=16, head_channels=1024,
        operation=BottleneckOperation),

    'AFF-ResNeXt-47-32x4d': update_params(
        small_basic_params,
        layers=make_resnet_layers([5, 5, 5], 4, 32, 64),
        stem_channels=16, head_channels=1024,
        operation=BottleneckOperation, affmodule=True),

    'ResNeSt-47-2s1x64d': update_params(
        small_basic_params,
        layers=make_resnest_layers([5, 5, 5], 64, 2, 1, 4),
        stem_channels=16, head_channels=1024,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=SplitAttentionOperation),

    'ResNeSt-47-2s1x128d': update_params(
        small_basic_params,
        layers=make_resnest_layers([5, 5, 5], 128, 2, 1, 2),
        stem_channels=16, head_channels=1024,
        stem=TweakedLargeStem, downsample=TweakedDownsample,
        operation=SplitAttentionOperation),

    'PyramidNet-110-a48': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 48, 1, 1),
        stem_channels=16, head_channels=64,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBasicOperation),

    'PyramidNet-110-a270': update_params(
        small_basic_params,
        layers=make_pyramid_layers([18, 18, 18], 16, 270, 1, 1),
        stem_channels=16, head_channels=286,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBasicOperation),

    'PyramidNet-200-a240': update_params(
        small_basic_params,
        layers=make_pyramid_layers([22, 22, 22], 16, 240, 1, 4),
        stem_channels=16, head_channels=1024,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBottleneckOperation),

    'PyramidNet-272-a200': update_params(
        small_basic_params,
        layers=make_pyramid_layers([30, 30, 30], 16, 200, 1, 4),
        stem_channels=16, head_channels=864,
        stem=PreActSmallStem, head=PreActHead,
        block=PreActBlock, downsample=AverageDownsample,
        operation=SingleActBottleneckOperation),
}

size256_models = {}
size256_models.update(large_models)
size256_models.update(dense_models(size256_models))
size256_models.update(skip_models(size256_models))

size64_models = {}
size64_models.update(update_models(
    {n: m for n, m in large_models.items() if m['stem'] == BasicLargeStem},
    stem=BasicSmallStem))
size64_models.update(dense_models(size64_models))
size64_models.update(skip_models(size64_models))

size32_models = {}
size32_models.update(small_models)
size32_models.update(dense_models(size32_models))
size32_models.update(skip_models(size32_models))

PARAMETERS = {
    'imagenet': update_models(size256_models, num_classes=1000),
    'tinyimagenet': update_models(size64_models, num_classes=200),
    'cifar100': update_models(size32_models, num_classes=100),
    'cifar10': update_models(size32_models, num_classes=10),
}
