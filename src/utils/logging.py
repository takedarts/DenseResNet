#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
import warnings


def setup_logging(debug):
    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] '
                                  '%(message)s (%(module)s.%(funcName)s:%(lineno)s)',
                                  '%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    warnings.filterwarnings('ignore', message='Corrupt EXIF data.  Expecting to read 4 bytes but only got 0.')
