import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, cat
from abc import ABCMeta, abstractmethod
from .error import check_length, try_convert, raise_error

import os

__all__ = [
    'DLDataset',
    'VectorCategoricalDataset',
    'VectorRegressionDataset',
    'PyTorchNativeDataset',
    'ImageCategoricalDataset',
    'EmptyDataset'
]


def load_images(folder_name):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    images = []
    for filename in os.listdir(folder_name):
        image = Image.open(filename)
        images.append(to_tensor(image))

    return torch.Tensor(images)


class DLDataset(Dataset, metaclass=ABCMeta):

    def get_data(self):
        features, targets = None, None
        loader = DataLoader(self, batch_size=128)
        for idx, (feature, target) in enumerate(loader):
            if idx == 0:
                features = feature
                targets = target
            else:
                features = cat([features, feature], dim=0)
                targets = cat([targets, target], dim=0)

        return Tensor(features), Tensor(targets)

    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def index_of(self, idx):
        pass

    def __len__(self):
        return self.get_length()

    def __getitem__(self, idx):
        return self.index_of(idx)


class EmptyDataset(DLDataset):
    def __init__(self):
        super().__init__()

    def index_of(self, idx):
        return None

    def get_length(self):
        return 0


class PyTorchNativeDataset(DLDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def index_of(self, idx):
        return self.dataset[idx]

    def get_length(self):
        return len(self.dataset)


class VectorCategoricalDataset(DLDataset):
    def __init__(self, features, targets):

        features = try_convert(features, Tensor, 'features', 'Tensor')
        targets = try_convert(targets, Tensor, 'targets', 'Tensor').long()

        check_length(features, targets, 'features', 'targets')

        try:
            features = features.reshape(features.shape[0], -1)
            targets = targets.reshape(targets.shape[0])
        except:
            raise_error('reshape features and targets',
                        f'features shape({features.shape}) or targets shape({targets.shape}) cannot reshape to [data.shape[0], -1]')

        self.features = features
        self.targets = targets

    def get_length(self):
        return len(self.features)

    def index_of(self, idx):
        return self.features[idx], self.targets[idx]


class ImageCategoricalDataset(DLDataset):
    def __init__(self, images, labels):

        images = try_convert(images, Tensor, 'features', 'Tensor')
        labels = try_convert(labels, Tensor, 'targets', 'Tensor').long()

        check_length(images, labels, 'images', 'labels')

        try:
            image_shape = images.shape
            if len(image_shape) == 3:
                # [batch, w, h] -> [batch, 1, w, h]
                images = images.reshape(image_shape[0], 1, image_shape[1], image_shape[2])

            labels = labels.reshape(-1)
        except:
            raise_error('reshape features and targets',
                        f'features shape({images.shape}) or labels shape({labels.shape}) cannot reshape')

        self.features = images
        self.targets = labels

    def get_length(self):
        return len(self.features)

    def index_of(self, idx):
        return self.features[idx], self.targets[idx]


class VectorRegressionDataset(DLDataset):
    def __init__(self, features, targets):

        features = try_convert(features, Tensor, 'features', 'Tensor')
        targets = try_convert(targets, Tensor, 'targets', 'Tensor')

        check_length(features, targets, 'features', 'targets')

        try:
            features = features.reshape(features.shape[0], -1)
            targets = targets.reshape(targets.shape[0], -1)
        except:
            raise_error('reshape features and targets',
                        f'features shape({features.shape}) or targets shape({targets.shape}) cannot reshape to [data.shape[0], -1]')

        self.features = features
        self.targets = targets

    def get_length(self):
        return len(self.features)

    def index_of(self, idx):
        return self.features[idx], self.targets[idx]


class ImageMaskedDataset(DLDataset):
    def __init__(self, source_image_folder, masked_image_folder):
        source_images = load_images(source_image_folder)
        masked_images = load_images(masked_image_folder)
        self.source_images = source_images
        self.masked_images = masked_images

        check_length(len(source_images), len(masked_images), 'source images', 'masked images')

    def index_of(self, idx):
        return self.source_images[idx], self.masked_images[idx]

    def get_length(self):
        return len(self.source_images)
