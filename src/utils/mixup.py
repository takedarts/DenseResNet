import torch
import numpy


def mixup(images, targets, prob, alpha=1.0):
    if numpy.random.rand() >= prob:
        return images, targets

    lam = numpy.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.shape[0], device=images.device)

    # generate mixed images
    images = lam * images + (1 - lam) * images[rand_index, :]

    # generate mixed targets
    targets = targets * lam + targets[rand_index] * (1 - lam)

    return images, targets
