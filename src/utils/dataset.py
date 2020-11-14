import torch.utils.data
import torchvision.datasets
import torchvision.transforms

import numpy
import os

from .autoaug import ImageNetPolicy, CIFAR10Policy


def load_dataset(dataset_name, data_dir, crop_size, train, stdaug,
                 autoaugment=False, random_erasing_prob=0.0, random_erasing_type='random'):
    if dataset_name == 'imagenet':
        return ImagenetDataset(
            os.path.join(data_dir, 'imagenet'), 1000, crop_size,
            train, stdaug, autoaugment, random_erasing_prob, random_erasing_type)
    elif dataset_name == 'cifar10':
        return Cifar10Dataset(
            os.path.join(data_dir, 'cifar'),
            train, stdaug, autoaugment, random_erasing_prob, random_erasing_type)
    elif dataset_name == 'cifar100':
        return Cifar100Dataset(
            os.path.join(data_dir, 'cifar'),
            train, stdaug, autoaugment, random_erasing_prob, random_erasing_type)
    else:
        raise Exception(f'unsuppoted dataset: {dataset_name}')


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3))
        rgb = rgb.mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class ImagenetDataset(torchvision.datasets.ImageFolder):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    LIGHTING = {
        'alphastd': 0.1,
        'eigval': [0.2175, 0.0188, 0.0045],
        'eigvec': [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]]
    }

    def __init__(self, path, num_classes, crop_size, train, stdaug,
                 autoaugment, random_erasing_prob, random_erasing_type):
        if train:
            path = os.path.join(path, 'train')
        else:
            path = os.path.join(path, 'valid')

        if stdaug:
            transforms = [
                torchvision.transforms.RandomResizedCrop(crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                torchvision.transforms.ToTensor(),
                Lighting(**self.LIGHTING),
                torchvision.transforms.Normalize(mean=self.MEAN, std=self.STD)]
        else:
            transforms = [
                torchvision.transforms.Resize(round(crop_size / 224 * 256)),
                torchvision.transforms.CenterCrop(crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=self.MEAN, std=self.STD)]

        if stdaug and autoaugment:
            transforms.insert(2, ImageNetPolicy())

        if random_erasing_prob != 0:
            value = 0 if random_erasing_type == 'zero' else 'random'
            transforms.append(torchvision.transforms.RandomErasing(
                p=random_erasing_prob, value=value))

        super().__init__(path, torchvision.transforms.Compose(transforms))
        self.num_classes = num_classes

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        label_array = numpy.zeros(self.num_classes, dtype=numpy.float32)
        label_array[label] = 1

        return image, label_array


class CifarDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, num_classes, mean, std, stdaug,
                 autoaugment, random_erasing_prob, random_erasing_type):
        super().__init__()
        if stdaug:
            transforms = [
                torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)]
        else:
            transforms = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std)]

        if stdaug and autoaugment:
            transforms.insert(2, CIFAR10Policy())

        if random_erasing_prob != 0:
            value = 0 if random_erasing_type == 'zero' else 'random'
            transforms.append(torchvision.transforms.RandomErasing(
                p=random_erasing_prob, value=value))

        self.transforms = torchvision.transforms.Compose(transforms)
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transforms(image)
        label_array = numpy.zeros(self.num_classes, dtype=numpy.float32)
        label_array[label] = 1

        return image, label_array


class Cifar10Dataset(CifarDataset):
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]

    def __init__(self, path, train, stdaug, autoaugment, random_erasing_prob, random_erasing_type):
        super().__init__(
            torchvision.datasets.CIFAR10(path, download=True, train=train), num_classes=10,
            mean=self.MEAN, std=self.STD, stdaug=stdaug, autoaugment=autoaugment,
            random_erasing_prob=random_erasing_prob, random_erasing_type=random_erasing_type)


class Cifar100Dataset(CifarDataset):
    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]

    def __init__(self, path, train, stdaug, autoaugment, random_erasing_prob, random_erasing_type):
        super().__init__(
            torchvision.datasets.CIFAR100(path, download=True, train=train), num_classes=100,
            mean=self.MEAN, std=self.STD, stdaug=stdaug, autoaugment=autoaugment,
            random_erasing_prob=random_erasing_prob, random_erasing_type=random_erasing_type)
