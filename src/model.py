import models
import utils

import logging
import argparse

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Show a list of models.', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('dataset', choices=list(models.PARAMETERS.keys()), help='Dataset name.')
parser.add_argument('--debug', action='store_true', default=False, help='Debug mode.')


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    for name in models.PARAMETERS[args.dataset].keys():
        print(name)


if __name__ == '__main__':
    main()
