#!/usr/bin/env python
# -*- coding: utf-8 -*-
import utils

import logging
import os
import argparse
import tarfile
import scipy.io
import zipfile

LOGGER = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')
DATASETS = {
    'imagenet': lambda: prepare_imagenet(),
    'tinyimagenet': lambda: prepare_tinyimagenet()}

parser = argparse.ArgumentParser(description='prepare a dataset')
parser.add_argument('name', choices=DATASETS.keys(), help='dataset name')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')


def prepare_imagenet():
    data_dir = os.path.join(DATA_DIR, 'imagenet')
    devkit_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12.tar.gz')
    train_file = os.path.join(data_dir, 'ILSVRC2012_img_train.tar')
    valid_file = os.path.join(data_dir, 'ILSVRC2012_img_val.tar')

    # read meta data
    meta_name = 'ILSVRC2012_devkit_t12/data/meta.mat'
    valid_name = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

    with tarfile.open(devkit_file, 'r') as reader:
        meta = scipy.io.loadmat(reader.extractfile(meta_name))
        valid = reader.extractfile(valid_name).read()

    wnid_labels = {r[0]['WNID'][0]: int(r[0]['ILSVRC2012_ID'][0, 0]) - 1
                   for r in meta['synsets'] if int(r[0]['ILSVRC2012_ID'][0, 0]) <= 1000}
    valid_labels = [int(s) - 1 for s in valid.decode('utf-8').split('\n') if len(s) != 0]

    # make image directories
    for i in range(1000):
        os.makedirs(os.path.join(data_dir, 'train', f'{i:03d}'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'valid', f'{i:03d}'), exist_ok=True)

    # copy train images
    with tarfile.open(train_file, 'r') as archive_reader:
        for archive in archive_reader:
            LOGGER.info(
                'make train images: %s (%d)', archive.name[:-4], wnid_labels[archive.name[:-4]])

            with tarfile.open(fileobj=archive_reader.extractfile(archive), mode='r') as reader:
                for idx, info in enumerate(reader):
                    path = os.path.join(
                        data_dir, 'train', f'{wnid_labels[archive.name[:-4]]:03d}', f'{idx}.jpg')

                    with open(path, 'wb') as writer:
                        writer.write(reader.extractfile(info).read())

    # copy validation images
    file_idxs = [0] * 1000

    LOGGER.info('make validation images')

    with tarfile.open(valid_file, 'r') as reader:
        labels = {n: v for n, v in zip(sorted(reader.getnames()), valid_labels)}

        for info in reader:
            label = labels[info.name]
            path = os.path.join(data_dir, 'valid', f'{label:03d}', f'{file_idxs[label]}.jpg')

            with open(path, 'wb') as writer:
                writer.write(reader.extractfile(info).read())

            file_idxs[label] += 1


def prepare_tinyimagenet():
    data_dir = os.path.join(DATA_DIR, 'tinyimagenet')
    data_file = os.path.join(data_dir, 'tiny-imagenet-200.zip')

    with zipfile.ZipFile(data_file, 'r') as reader:
        # read labels
        wnid_labels = reader.read('tiny-imagenet-200/wnids.txt')
        wnid_labels = wnid_labels.decode('utf-8').split('\n')
        wnid_labels = {n: i for i, n in enumerate(wnid_labels) if len(n) != 0}

        # make image directories
        for i in range(200):
            os.makedirs(os.path.join(data_dir, 'train', f'{i:03d}'), exist_ok=True)
            os.makedirs(os.path.join(data_dir, 'valid', f'{i:03d}'), exist_ok=True)

        # copy train images
        LOGGER.info('make train images')

        for name in reader.namelist():
            if not name.startswith('tiny-imagenet-200/train/') or not name.endswith('.JPEG'):
                continue

            wnid, file = name.split('/')[-1].split('_')
            path = os.path.join(data_dir, 'train', f'{wnid_labels[wnid]:03d}', file)

            with open(path, 'wb') as writer:
                writer.write(reader.read(name))

        # copy validation images
        LOGGER.info('make validation images')

        annotations = reader.read('tiny-imagenet-200/val/val_annotations.txt')
        annotations = annotations.decode('utf-8').split('\n')
        annotations = [v.split()[:2] for v in annotations if len(v) != 0]

        for name, wnid in annotations:
            file = name.split('_')[1]
            path = os.path.join(data_dir, 'valid', f'{wnid_labels[wnid]:03d}', file)

            with open(path, 'wb') as writer:
                writer.write(reader.read('tiny-imagenet-200/val/images/{}'.format(name)))


def main():
    args = parser.parse_args()
    utils.setup_logging(args.debug)

    DATASETS[args.name]()


if __name__ == '__main__':
    main()
