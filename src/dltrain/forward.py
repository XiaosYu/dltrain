from abc import ABCMeta, abstractmethod
from typing import Tuple

from .evaluation import Evaluation

from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor

__all__ = [
    'Forward',
    'SimpleForward'
]

class Forward(metaclass=ABCMeta):
    @abstractmethod
    def train(self, model: Module, optimizer: Optimizer, criterion: Module, x: Tensor, y: Tensor, *args, **kwargs) -> \
            Tuple[float, Tensor]:
        pass

    @abstractmethod
    def eval(self, model: Module, criterion: Module, x: Tensor, y: Tensor, *args, **kwargs) -> Tuple[float, Tensor]:
        pass

    def forward(self, model: Module, criterion: Module, x: Tensor, y: Tensor, eval: bool,
                evaluation: Evaluation = None, optimizer: Optimizer = None, *args, **kwargs):
        if eval:
            loss, out = self.eval(model, criterion, x, y)
        else:
            loss, out = self.train(model, optimizer, criterion, x, y)

        if evaluation is not None:
            evaluation.append(out)

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SimpleForward(Forward):

    def eval(self, model: Module, criterion: Module, x: Tensor, y: Tensor, *args, **kwargs) -> Tuple[float, Tensor]:
        out = model(x)
        loss = criterion(out, y)
        return loss, out

    def train(self, model: Module, optimizer: Optimizer, criterion: Module, x: Tensor, y: Tensor, *args, **kwargs) -> \
            Tuple[float, Tensor]:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss, out
