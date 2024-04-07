from abc import ABCMeta, abstractmethod

from .dataset import DLDataset, PyTorchNativeDataset
from torch.utils.data import random_split


__all__ = [
    'Delineator',
    'TrainEvalSetDelineator',
    'RandomSplitDelineator'
]

class Delineator(metaclass=ABCMeta):
    @abstractmethod
    def get_train_set(self) -> DLDataset:
        pass

    @abstractmethod
    def get_eval_set(self) -> DLDataset:
        pass


class TrainEvalSetDelineator(Delineator):
    def __init__(self, train_set: DLDataset, eval_set: DLDataset):
        self.train_set = train_set
        self.eval_set = eval_set

    def get_eval_set(self) -> DLDataset:
        return self.eval_set

    def get_train_set(self) -> DLDataset:
        return self.train_set


class RandomSplitDelineator(Delineator):
    def __init__(self, dataset: DLDataset, train_rate=0.8, eval_rate=0.2):
        train_set, eval_set = random_split(dataset, [train_rate, eval_rate])
        self.train_set = PyTorchNativeDataset(train_set)
        self.eval_set = PyTorchNativeDataset(eval_set)

    def get_eval_set(self) -> DLDataset:
        return self.eval_set

    def get_train_set(self) -> DLDataset:
        return self.train_set
