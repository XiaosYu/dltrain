import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn

from .forward import InjectForward
from abc import ABCMeta, abstractmethod
import seaborn as sns


class InjectBase(metaclass=ABCMeta):
    def __init__(self):
        self.property: str = None

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()


class GradientAcquisition(InjectBase):
    def __init__(self, folder):
        super().__init__()

        self.property = 'after_backward'
        self.folder = folder

    def call(self, *args, **kwargs):
        if 'model' in kwargs:
            model: torch.nn.Module = kwargs['model']

            grads = [parameters.grad.cpu().numpy().reshape(-1).mean() for parameters in model.parameters() if
                     parameters.grad is not None]

            plt.cla()

            data = pd.DataFrame({
                'layer': range(len(grads)),
                'grads': grads,
            })

            sns.barplot(x='layer', y='grads', data=data, dodge=True)
            plt.savefig(f"{self.folder}/{kwargs['epoch']}_{kwargs['idx']}.png", dpi=100)

            del data
