from typing import Union

from .core import Wizard
from ..checkpoint import CheckPoint

import torch


class BaseWizard(Wizard):
    def __init__(self):
        self._batch_size = 16
        self._seed = 3407
        self._start_checkpoint: str = None
        self._save_checkpoint: bool = False
        self._epochs: int = 10
        self._device: torch.device = torch.device('cpu')

    def use_batch_size(self, bs):
        self._batch_size = bs
        return self

    def use_seed(self, seed):
        self._seed = seed
        return self

    def set_checkpoint(self, checkpoint: bool = False):
        self._save_checkpoint = checkpoint
        return self

    def use_checkpoint(self, checkpoint: str) -> CheckPoint:
        self._start_checkpoint = torch.load(checkpoint)
        return self._start_checkpoint

    def use_epoch(self, epochs: int = 10):
        self._epochs = epochs
        return self

    def use_device(self, device: Union[torch.device, str]):
        if isinstance(device, str):
            self._device = torch.device(device)
        elif isinstance(device, torch.device):
            self._device = device

        return self
