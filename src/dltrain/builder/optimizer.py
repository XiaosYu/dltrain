from torch.optim import SGD, Adam, AdamW, Optimizer
from abc import ABCMeta, abstractmethod
from typing import Type

__all__ = [
    'OptimizerBuilder',
    'AdamBuilder',
    'AdamWBuilder',
    'SgdBuilder'
]

class OptimizerBuilder(metaclass=ABCMeta):
    def __init__(self):
        self.parameters = {}

    @abstractmethod
    def get_type(self) -> Type[Optimizer]:
        pass

    def get_parameters(self):
        return self.parameters

    def set_lr(self, lr: float):
        self.parameters['lr'] = lr
        return self


class SgdBuilder(OptimizerBuilder):
    def get_type(self) -> Type[Optimizer]:
        return SGD

class AdamBuilder(OptimizerBuilder):
    def get_type(self) -> Type[Optimizer]:
        return Adam

class AdamWBuilder(OptimizerBuilder):
    def get_type(self) -> Type[Optimizer]:
        return AdamW
