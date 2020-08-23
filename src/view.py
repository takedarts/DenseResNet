import models
import utils
import torch
import logging
import argparse
import time

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='view data of a snapshot')
parser.add_argument('file', help='file name')
parser.add_argument('--model', action='store_true', default=False, help='view model')
parser.add_argument('--log', action='store_true', default=False, help='view all logs')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    # load file
    snapshot = torch.load(args.file, map_location=lambda s, _: s)
    params = snapshot['params']

    # view model
    models.CONFIG.semodule_reduction = params.semodule_reduction
    models.CONFIG.gate_reduction = params.gate_reduction
    models.CONFIG.gate_connections = params.gate_connections
    model = models.create_model(params.dataset, params.model)

    print('[model]')

    if args.model:
        print(model)

    print('parameters={:,d}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if 'flops' in snapshot:
        print('flops={:,d}'.format(snapshot['flops']))

    # view parameters
    print()
    print('[parameters]')

    for k, v in params.__dict__.items():
        print(f'{k}={v}')

    # view log
    print()
    print('[log]')

    if len(snapshot['log']) != 0:
        logs = snapshot['log'] if args.log else snapshot['log'][-1:]

        for log in logs:
            print('time={}, epoch={epoch:d}, learning-rate={learning-rate:6f},'.format(
                time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime(log['time'])), **log))

            for c in ('train', 'valid'):
                values = [(m, log[f'{c}-{m}']) for m in ('loss', 'accuracy1', 'accuracy5')]
                print('{}: {}'.format(c, ', '.join(f'{k}={v:.6f}' for k, v in values)))
    else:
        print('no log')

    # view result
    print()
    print('[result]')

    if 'result' in snapshot:
        results = snapshot['result']

        print('loss(mCE)={loss:.6f}'.format(**results))
        print('accuracy(top-1)={accuracy1:.6f}'.format(**results))
        print('accuracy(top-5)={accuracy5:.6f}'.format(**results))
        print('throughput={throughput:.2f} [images/sec]'.format(**results))
    else:
        print('no result')


if __name__ == '__main__':
    main()
