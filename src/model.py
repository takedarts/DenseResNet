#!/usr/bin/env python
# -*- coding: utf-8 -*-
import models
import utils

import logging
import argparse

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='show a list of models')
parser.add_argument('dataset', help='dataset name')
parser.add_argument('model', nargs='?', default=None, help='model name')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


def print_list(args):
    for name in models.PARAMETERS[args.dataset].keys():
        print(name)


def print_model(args):
    params = models.PARAMETERS[args.dataset][args.model]
    components = ('stem', 'classifier', 'block', 'operation', 'downsample', 'shortcut')

    print('parameters:')
    for key, value in params.items():
        if key != 'layers' and key not in components:
            print(f'  {key}={value}')

    print('components:')
    for comp in components:
        print('  {}={}'.format(comp, params[comp].__name__))

    print('layers:')
    for c, s in params['layers']:
        print(f'  channels={c}, stride={s}')


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    if args.model is not None:
        print_model(args)
    else:
        print_list(args)


if __name__ == '__main__':
    main()
