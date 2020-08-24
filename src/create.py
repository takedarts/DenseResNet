import models
import utils

import torch

import argparse
import logging
import copy

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Create a model file', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('file', type=str, help='File name.')
parser.add_argument('dataset', type=str, help='Dataset name.')
parser.add_argument('model', type=str, help='Model name.')
parser.add_argument('--config', type=str, default=None, help='Configuration file.')

for name, value, desc in utils.OPTIONS:
    kwargs = {
        'default': value,
        'help': f'{desc} (default:{value})'}

    if isinstance(value, bool):
        kwargs['type'] = utils.arg_type_bool
    elif isinstance(value, int):
        kwargs['type'] = int
    elif isinstance(value, float):
        kwargs['type'] = float
    else:
        kwargs['type'] = str

    parser.add_argument(f'--{name}', **kwargs)

parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    # defaults
    if args.config is not None:
        with open(args.config, 'r') as reader:
            configs = [v.split(':') for v in reader if len(v) != 0]
        configs = {k.replace('-', '_').strip(): v.strip() for k, v in configs}
        parser.set_defaults(**configs)

    args = parser.parse_args()

    # settings
    utils.random_seed(args.seed)
    models.CONFIG.semodule_reduction = args.semodule_reduction
    models.CONFIG.gate_reduction = args.gate_reduction
    models.CONFIG.gate_connections = args.gate_connections
    models.CONFIG.dropblock_size = args.dropblock_size

    # model
    model = models.create_model(
        args.dataset, args.model,
        dropout=args.dropblock_prob,
        shakedrop=args.shakedrop_prob,
        sigaug=args.signal_augment)

    LOGGER.debug(model)
    LOGGER.info('number of parameters={:,d}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer
    optimizer = utils.create_optimizer(
        model, args.train_lr, args.train_wdecay, args.train_bdecay)
    tuner = utils.create_optimizer(
        model.classifier, args.tune_lr, args.tune_wdecay)

    scheduler = utils.CosineAnnealingLR(optimizer, args.train_epoch, args.train_warmup)

    LOGGER.debug('optimizer:\n%s', optimizer)
    LOGGER.debug('tuner:\n%s', tuner)
    LOGGER.debug('scheduler:\n%s', scheduler.state_dict())

    # save
    params = copy.copy(args)
    del params.file
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
