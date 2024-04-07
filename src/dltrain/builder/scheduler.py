from abc import ABCMeta, abstractmethod
from typing import Type

from torch.optim.lr_scheduler import LRScheduler, StepLR

__all__ = [
    'SchedulerBuilder',
    'StepLrBuilder'
]

class SchedulerBuilder(metaclass=ABCMeta):
    def __init__(self):
        self.parameters = {}

    @abstractmethod
    def get_type(self) -> Type[LRScheduler]:
        pass

    def get_parameters(self):
        return self.parameters


class StepLrBuilder(SchedulerBuilder):
    def get_type(self) -> Type[LRScheduler]:
        return StepLR

    def set_step_size(self, step_size):
        self.parameters['step_size'] = step_size
        return self

    def set_gamma(self, gamma):
        self.parameters['gamma'] = gamma
        return self
