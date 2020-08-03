#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy


def rand_bbox(width, height, lam):
    cut_rat = numpy.sqrt(1 - lam)
    cut_w = numpy.int(width * cut_rat)
    cut_h = numpy.int(height * cut_rat)

    cx = numpy.random.randint(width)
    cy = numpy.random.randint(height)

    bbx1 = numpy.clip(cx - cut_w // 2, 0, width)
    bby1 = numpy.clip(cy - cut_h // 2, 0, height)
    bbx2 = numpy.clip(cx + cut_w // 2, 0, width)
    bby2 = numpy.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, targets, prob, beta=1.0):
    if numpy.random.rand() >= prob:
        return images, targets

    lam = numpy.random.beta(beta, beta)
    rand_index = torch.randperm(images.shape[0], device=images.device)  # @UndefinedVariable

    # generate mixed images
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.shape[3], images.shape[2], lam)
    images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]

    # generate mixed targets
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[2] * images.shape[3]))
    targets = targets * lam + targets[rand_index] * (1 - lam)

    return images, targets
