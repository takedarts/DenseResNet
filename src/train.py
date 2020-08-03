#!/usr/bin/env python
# -*- coding: utf-8 -*-
import models.modules
import utils

import torch.cuda
import torch.utils.data
import torch.optim as optim

import argparse
import logging
import time
import os

LOGGER = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(description='train a model')
parser.add_argument('file', help='file name')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers on data loader')
parser.add_argument('--opt', type=int, default=0, choices=range(4),
                    help='optimization level')
parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))), default=[],
                    help='device id of gpu')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


def update_dropblock(module, prob):
    if isinstance(module, models.modules.DropBlock):
        module.drop_prob = prob
    else:
        for child in module.children():
            update_dropblock(child, prob)


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    # load file
    snapshot = torch.load(args.file, map_location=lambda s, _: s)
    params = snapshot['params']

    # view properties
    for k, v in params.__dict__.items():
        LOGGER.info(f'parameter: {k}={v}')

    # settings
    utils.random_seed(params.seed)
    models.CONFIG.semodule_reduction = params.semodule_reduction
    models.CONFIG.gate_reduction = params.gate_reduction
    models.CONFIG.gate_connections = params.gate_connections

    if torch.cuda.device_count() != 0:
        torch.torch.backends.cudnn.benchmark = True
        torch.torch.backends.cudnn.enabled = True

    # model
    kwargs = {
        'dropout': params.dropout_prob,
        'shakedrop': params.shakedrop_prob,
        'sigaug': params.sigaugment}
    model = models.create_model(params.dataset, params.model, **kwargs)
    model.load_state_dict(snapshot['model'])

    LOGGER.debug(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=0)
    scheduler = utils.CosineAnnealingLR(optimizer)

    optimizer.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])

    LOGGER.debug('optimizer:\n%s', optimizer)
    LOGGER.debug('scheduler:\n%s', scheduler.state_dict())

    # dataset
    train_dataset = utils.load_dataset(
        params.dataset, DATA_DIR, params.train_crop,
        train=True, stdaug=True, autoaug=params.autoaugment)
    valid_dataset = utils.load_dataset(
        params.dataset, DATA_DIR, params.tune_crop,
        train=False, stdaug=False, autoaug=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.train_batch,
        shuffle=True, pin_memory=False, num_workers=args.workers,
        worker_init_fn=lambda x: utils.random_seed(params.seed + x))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=params.train_batch,
        shuffle=False, pin_memory=False, num_workers=args.workers)

    # trainer
    trainer = utils.Trainer(model, optimizer, args.gpu, args.opt)

    # train
    for epoch in range(len(snapshot['log']), params.train_epoch):
        # logs
        log = {}
        log['time'] = time.time()
        log['epoch'] = epoch + 1
        log['learning-rate'] = float(optimizer.param_groups[0]['lr'])
        LOGGER.info('epoch=%(epoch)d, learning-rate=%(learning-rate).6f', log)

        # dropblock
        update_dropblock(
            model, params.dropblock_prob * epoch / max(params.train_epoch - 1, 1))

        # train-train
        trainer.train(
            train_loader, label_smooth=params.label_smooth,
            cutmix_prob=params.cutmix_prob, cutmix_beta=params.cutmix_beta,
            mixup_prob=params.mixup_prob, mixup_beta=params.mixup_beta)
        log.update({f'train-{k}': v for k, v in trainer.get_status().items()})
        LOGGER.info('train: %s', trainer)

        # valid-valid
        trainer.validate(valid_loader)
        log.update({f'valid-{k}': v for k, v in trainer.get_status().items()})
        LOGGER.info('valid: %s', trainer)

        # update status
        scheduler.step()

        # save
        snapshot['log'].append(log)
        snapshot['model'] = model.state_dict()
        snapshot['optimizer'] = optimizer.state_dict()
        snapshot['scheduler'] = scheduler.state_dict()

        torch.save(snapshot, args.file)


if __name__ == '__main__':
    main()
