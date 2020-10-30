import models.modules
import utils

import torch.cuda
import torch.utils.data

import argparse
import logging
import time
import os

LOGGER = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')

parser = argparse.ArgumentParser(
    description='Train a model', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('file', type=str, help='File name.')
parser.add_argument('--workers', type=int, default=4, help='Number of workers on data loader.')
parser.add_argument('--opt', type=int, default=0, choices=range(4), help='Optimization level.')
parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))), default=[], help='GPU IDs.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def update_dropblock(module, drop_prob):
    if isinstance(module, models.modules.DropBlock):
        module.drop_prob = drop_prob
    else:
        for child in module.children():
            update_dropblock(child, drop_prob)


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
    models.CONFIG.load(params)

    if torch.cuda.device_count() != 0:
        torch.torch.backends.cudnn.benchmark = True
        torch.torch.backends.cudnn.enabled = True

    # model
    model = models.create_model(
        params.dataset, params.model,
        dropout=params.dropout_prob,
        shakedrop=params.shakedrop_prob,
        signalaugment=params.signalaugment)
    model.load_state_dict(snapshot['model'])

    LOGGER.debug(model)

    # optimizer
    optimizer = utils.create_optimizer(model, 0, 0, params.train_bdecay)
    scheduler = utils.CosineAnnealingLR(optimizer)

    optimizer.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])

    LOGGER.debug('optimizer:\n%s', optimizer)
    LOGGER.debug('scheduler:\n%s', scheduler.state_dict())

    # dataset
    train_dataset = utils.load_dataset(
        params.dataset, DATA_DIR, params.train_crop, train=True, stdaug=True,
        autoaugment=params.autoaugment,
        random_erasing_prob=params.randomerasing_prob,
        random_erasing_type=params.randomerasing_type)
    valid_dataset = utils.load_dataset(
        params.dataset, DATA_DIR, params.valid_crop, train=False, stdaug=False)

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
            train_loader, label_smooth=params.labelsmooth,
            cutmix_prob=params.cutmix_prob, cutmix_alpha=params.cutmix_alpha,
            mixup_prob=params.mixup_prob, mixup_alpha=params.mixup_alpha)
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

        torch.save(snapshot, f'{args.file}.temp')
        os.rename(f'{args.file}.temp', args.file)


if __name__ == '__main__':
    main()
