#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy


def mixup(images, targets, prob, beta=1.0):
    if numpy.random.rand() >= prob:
        return images, targets

    lam = numpy.random.beta(beta, beta)
    rand_index = torch.randperm(images.shape[0], device=images.device)  # @UndefinedVariable

    # generate mixed images
    images = lam * images + (1 - lam) * images[rand_index, :]

    # generate mixed targets
    targets = targets * lam + targets[rand_index] * (1 - lam)

    return images, targets
