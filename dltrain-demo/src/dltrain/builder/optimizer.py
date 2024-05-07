from torch.optim import SGD, Adam, AdamW, Optimizer
from .core import Wizard

__all__ = [
    'OptimizerWizard'
]


class OptimizerWizard(Wizard):
    def __init__(self):
        self._type = None
        self._parameters = {}

        self.use_sgd()

    def use_optimizer(self, optimizer: type[Optimizer], **kwargs):
        self._type = optimizer
        self._parameters = kwargs
        return self

    def use_sgd(self, lr=1e-2, momentum=0, dampening=0, weight_decay=0):
        return self.use_optimizer(SGD, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)

    def use_adam(self, lr=1e-2, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0,
                 amsgrad: bool = False):
        return self.use_optimizer(Adam, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def use_adamw(self, lr=1e-2, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0,
                 amsgrad: bool = False):
        return self.use_optimizer(AdamW, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)