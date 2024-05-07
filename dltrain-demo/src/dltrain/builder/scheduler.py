from .core import Wizard

from torch.optim.lr_scheduler import LRScheduler, StepLR

__all__ = [
    'SchedulerWizard'
]


class SchedulerWizard(Wizard):
    def __init__(self):
        self._type = None
        self._parameters = None

    def use_scheduler(self, scheduler: type[LRScheduler], **kwargs):
        self._type = scheduler
        self._parameters = kwargs
        return self

    def use_step_lr(self, step_size, gamma):
        return self.use_scheduler(StepLR, step_size=step_size, gamma=gamma)
