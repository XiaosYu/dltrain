from typing import Union

import numpy as np
import torch

from torch.utils.data import Dataset
from ..dataset import (PyTorchNativeDataset, ImageCategoricalDataset, VectorCategoricalDataset, DLDataset,
                       VectorRegressionDataset)


class DatasetWizard:
    @classmethod
    def use_dataset(cls, dataset: Union[DLDataset, Dataset]):
        return dataset if isinstance(dataset, DLDataset) else PyTorchNativeDataset(dataset)

    @classmethod
    def use_mnist(cls, root, train, download=False):
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        dataset = MNIST(root, train=train, transform=ToTensor(), download=download)
        return PyTorchNativeDataset(dataset)

    @classmethod
    def use_vector_regression(cls, features_vector: Union[torch.Tensor, np.ndarray],
                              targets_vector: Union[torch.Tensor, np.ndarray]):
        return VectorRegressionDataset(features_vector, targets_vector)

    @classmethod
    def use_vector_classify(cls, features_vector: Union[torch.Tensor, np.ndarray],
                            targets_vector: Union[torch.Tensor, np.ndarray]):
        return VectorCategoricalDataset(features_vector, targets_vector)

    @classmethod
    def use_moons(cls, n_samples=100, noise=None):
        from sklearn.datasets import make_moons

        x, y = make_moons(n_samples=n_samples, noise=noise)
        return cls.use_vector_classify(x, y)

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

    @classmethod
    def use_lfw(cls):
        from sklearn.datasets import fetch_lfw_people

        lfw = fetch_lfw_people()
        features, labels = lfw.images, lfw.target

        return ImageCategoricalDataset(features, labels)
