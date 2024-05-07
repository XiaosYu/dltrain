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
    def train(self, model: Module, optimizer: Optimizer, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
              *args, **kwargs) -> \
            Tuple[float, Tensor]:
        pass

    @abstractmethod
    def eval(self, model: Module, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
             *args, **kwargs) -> Tuple[float, Tensor]:
        pass

    def forward(self, model: Module, criterion: Module, x: Tensor, y: Tensor, eval: bool, epoch: int, idx: int,
                evaluation: Evaluation = None, optimizer: Optimizer = None, *args, **kwargs):
        if eval:
            loss, out = self.eval(model, criterion, x, y, epoch, idx)
        else:
            loss, out = self.train(model, optimizer, criterion, x, y, epoch, idx)

        if evaluation is not None:
            evaluation.append(out)

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SimpleForward(Forward):

    def eval(self, model: Module, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
             *args, **kwargs) -> Tuple[float, Tensor]:
        out = model(x)
        loss = criterion(out, y)
        return loss, out

    def train(self, model: Module, optimizer: Optimizer, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
              *args, **kwargs) -> \
            Tuple[float, Tensor]:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss, out


class InjectForward(Forward):
    def __init__(self):
        self.before_forward = None
        self.after_forward = None

        self.before_criterion = None
        self.after_criterion = None

        self.before_backward = None
        self.after_backward = None

        self.before_optimize = None
        self.after_optimize = None

        self.before_zero_grad = None
        self.after_zero_grad = None

    def bind(self, inject):
        self.__dict__[inject.property] = inject.call
        return self

    def eval(self, model: Module, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
             *args, **kwargs) -> Tuple[float, Tensor]:
        if self.before_forward is not None:
            self.before_forward(mode='eval', event='before_forward', model=model, criterion=criterion, x=x, y=y,
                                epoch=epoch, idx=idx, **kwargs)
        out = model(x)

        if self.after_forward is not None:
            self.before_forward(mode='eval', event='after_forward', model=model, criterion=criterion, x=x, y=y,
                                epoch=epoch, idx=idx, **kwargs)

        if self.before_criterion is not None:
            self.before_criterion(mode='eval', event='before_criterion', model=model, out=out, criterion=criterion,
                                  x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        loss = criterion(out, y)

        if self.after_criterion is not None:
            self.after_criterion(mode='eval', event='after_criterion', model=model, out=out, criterion=criterion,
                                 loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        return loss, out

    def train(self, model: Module, optimizer: Optimizer, criterion: Module, x: Tensor, y: Tensor, epoch: int, idx: int,
              *args, **kwargs) -> \
            Tuple[float, Tensor]:

        if self.before_forward is not None:
            self.before_forward(mode='train', event='before_forward', model=model, criterion=criterion, x=x, y=y,
                                epoch=epoch, idx=idx, **kwargs)
        out = model(x)

        if self.after_forward is not None:
            self.before_forward(mode='train', event='after_forward', model=model, criterion=criterion, x=x, y=y,
                                epoch=epoch, idx=idx, **kwargs)

        if self.before_criterion is not None:
            self.before_criterion(mode='train', event='before_criterion', model=model, out=out, criterion=criterion,
                                  x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        loss = criterion(out, y)

        if self.after_criterion is not None:
            self.after_criterion(mode='train', event='after_criterion', model=model, out=out, criterion=criterion,
                                 loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        if self.before_backward is not None:
            self.before_backward(mode='train', event='before_backward', model=model, out=out, criterion=criterion,
                                 loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        loss.backward()

        if self.after_backward is not None:
            self.after_backward(mode='train', event='after_backward', model=model, out=out, criterion=criterion,
                                loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        if self.before_optimize is not None:
            self.before_optimize(mode='train', event='before_optimize', model=model, out=out, criterion=criterion,
                                 loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        optimizer.step()

        if self.after_optimize is not None:
            self.after_optimize(mode='train', event='after_optimize', model=model, out=out, criterion=criterion,
                                loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        if self.before_zero_grad is not None:
            self.before_zero_grad(mode='train', event='before_zero_grad', model=model, out=out, criterion=criterion,
                                  loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        optimizer.zero_grad()

        if self.after_zero_grad is not None:
            self.after_zero_grad(mode='train', event='after_zero_grad', model=model, out=out, criterion=criterion,
                                 loss=loss, x=x, y=y, epoch=epoch, idx=idx, **kwargs)

        return loss, out
