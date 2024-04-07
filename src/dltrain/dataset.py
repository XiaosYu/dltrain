from torch.utils.data import Dataset, DataLoader
from torch import Tensor, cat
from abc import ABCMeta, abstractmethod

from .error import check_length, try_convert, raise_error

__all__ = [
    'DatasetWizard',
    'DLDataset',
    'VectorCategoricalDataset',
    'VectorRegressionDataset',
    'PyTorchNativeDataset'
]

class DatasetWizard:
    @classmethod
    def use_mnist(cls, root, train, download=False):
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        dataset = MNIST(root, train=train, transform=ToTensor(), download=download)
        return PyTorchNativeDataset(dataset)

    @classmethod
    def use_iris(cls):
        from sklearn.datasets import load_iris

        iris = load_iris()
        data = iris.data
        target = iris.target
        return VectorCategoricalDataset(data, target)

    @classmethod
    def use_image_folder(cls, root):
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import ToTensor

        dataset = ImageFolder(root, transform=ToTensor())
        return PyTorchNativeDataset(dataset)


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
