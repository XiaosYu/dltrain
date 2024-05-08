from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor

__all__ = [
    'Transform',
    'Container',
    'Resize',
    'Noise',
    'Standardize'
]

class Transform(metaclass=ABCMeta):
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data: Tensor):
        pass


class Noise(Transform):
    def __init__(self, rate=0.01):
        self.rate = rate

    def transform(self, data: Tensor):
        noise = torch.rand(data.shape).to(data.device) * self.rate
        return data + noise


class Container(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, data: Tensor):
        if self.transforms is None:
            return data

        for transform in self.transforms:
            data = transform(data)
        return data


class Resize(Transform):
    def __init__(self, size):
        from torchvision.transforms.functional_tensor import resize
        self.func = resize
        self.size = size

    def transform(self, data: Tensor):
        return self.func(data, size=self.size)


class Standardize(Transform):
    def transform(self, data: Tensor):
        min_ = data.min()
        max_ = data.max()
        data = (data - min_) / (max_ - min_)
        return data
