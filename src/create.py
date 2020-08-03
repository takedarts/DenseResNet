#!/usr/bin/env python
# -*- coding: utf-8 -*-
import models
import utils

import torch
import torch.optim as optim

import argparse
import logging
import copy

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='create a model')
parser.add_argument('file', help='file name')
parser.add_argument('dataset', help='dataset name')
parser.add_argument('model', help='model name')
parser.add_argument(
    '--config', type=str, default=None, help='config file of base paramteres')
parser.add_argument(
    '--base', type=str, default=None, help='model file of base paramteres')
parser.add_argument(
    '--train-batch', type=int, default=64, help='batch size for train')
parser.add_argument(
    '--train-crop', type=int, default=224, help='crop size for train')
parser.add_argument(
    '--train-epoch', type=int, default=300, help='number of epochs for train')
parser.add_argument(
    '--train-warmup', type=int, default=5, help='warm-up epochs for train')
parser.add_argument(
    '--train-lr', type=float, default=0.025, help='initial learning rate for train')
parser.add_argument(
    '--train-wdecay', type=float, default=0.0001, help='weight decay for train')
parser.add_argument(
    '--tune-batch', type=int, default=256, help='batch size for tuning')
parser.add_argument(
    '--tune-crop', type=int, default=224, help='crops size for tuning')
parser.add_argument(
    '--tune-epoch', type=int, default=60, help='number of epochs for tuning')
parser.add_argument(
    '--tune-lr', type=float, default=0.004, help='learning rate for tuning')
parser.add_argument(
    '--tune-wdecay', type=float, default=0.0001, help='weight decay for tuning')
parser.add_argument(
    '--cutmix-beta', type=float, default=1.0, help='beta parameter of cutmix')
parser.add_argument(
    '--cutmix-prob', type=float, default=0.0, help='probability of cutmix')
parser.add_argument(
    '--mixup-beta', type=float, default=1.0, help='beta parameter of mixup')
parser.add_argument(
    '--mixup-prob', type=float, default=0.0, help='probability of mixup')
parser.add_argument(
    '--autoaugment', action='store_true', default=False, help='use auto augmentation')
parser.add_argument(
    '--label-smooth', type=float, default=0.0, help='parmeter k of label smoothing')
parser.add_argument(
    '--dropout-prob', type=float, default=0.0, help='probability of dropoug')
parser.add_argument(
    '--shakedrop-prob', type=float, default=0.0, help='probability of shake-drop')
parser.add_argument(
    '--dropblock-prob', type=float, default=0.0, help='drop probability of dropblock')
parser.add_argument(
    '--dropblock-size', type=int, default=7, help='block size of dropblock')
parser.add_argument(
    '--stochdepth-prob', type=float, default=0.0, help='drop probability of stochasitic depth')
parser.add_argument(
    '--sigaugment', type=float, default=0.0, help='standard deviation of signal augmenation')
parser.add_argument(
    '--semodule-reduction', type=int, default=16, help='reduction rate of se-modules')
parser.add_argument(
    '--gate-reduction', type=int, default=8, help='reduction rate of shortcut gates')
parser.add_argument(
    '--gate-connections', type=int, default=4, help='number of connections into a shortcut gate')
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    # defaults
    if args.config is not None:
        with open(args.config, 'r') as reader:
            configs = [v.split(':') for v in reader if len(v) != 0]
        configs = {k.replace('-', '_').strip(): v.strip() for k, v in configs}
        parser.set_defaults(**configs)

    if args.base is not None:
        params = torch.load(args.base, map_location=lambda s, _: s)['params']
        parser.set_defaults(**{k: v for k, v in vars(params).items()})

    args = parser.parse_args()

    # settings
    utils.random_seed(args.seed)
    models.CONFIG.semodule_reduction = args.semodule_reduction
    models.CONFIG.gate_reduction = args.gate_reduction
    models.CONFIG.gate_connections = args.gate_connections
    models.CONFIG.dropblock_size = args.dropblock_size

    # model
    kwargs = {
        'dropout': args.dropout_prob,
        'shakedrop': args.shakedrop_prob,
        'sigaug': args.sigaugment}
    model = models.create_model(args.dataset, args.model, **kwargs)

    LOGGER.debug(model)
    LOGGER.info('number of parameters={:,d}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.train_lr,
        weight_decay=args.train_wdecay, momentum=0.9, nesterov=True)
    tuner = optim.SGD(
        model.classifier.parameters(), lr=args.tune_lr,
        weight_decay=args.tune_wdecay, momentum=0.9, nesterov=True)

    scheduler = utils.CosineAnnealingLR(optimizer, args.train_epoch, args.train_warmup)

    LOGGER.debug('optimizer:\n%s', optimizer)
    LOGGER.debug('tuner:\n%s', tuner)
    LOGGER.debug('scheduler:\n%s', scheduler.state_dict())

    # save
    params = copy.copy(args)
    del params.file
    del params.base
    del params.config
    del params.debug

    snapshot = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'tuner': tuner.state_dict(),
        'scheduler': scheduler.state_dict(),
        'params': params,
        'log': []}

    torch.save(snapshot, args.file)

    for k, v in params.__dict__.items():
        LOGGER.info('parameter: %s=%s', k, v)


if __name__ == '__main__':
    main()
