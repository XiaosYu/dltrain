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
        self._log_level: int = 0

    def use_batch_size(self, bs):
        self._batch_size = bs
        return self

    def use_seed(self, seed):
        self._seed = seed
        return self

    def use_log_level(self, level: int):
        self._log_level = level
        return self

    def use_log_loss(self):
        pass

    def use_checkpoint(self):
        self._save_checkpoint = True
        return self

    def use_checkpoint_path(self, checkpoint: str) -> CheckPoint:
        self._start_checkpoint = torch.load(checkpoint)
        return self._start_checkpoint

    def use_epoch(self, epochs: int = 10):
        self._epochs = epochs
        return self

    def use_device(self, device: Union[torch.device, str] = 'cuda'):
        if isinstance(device, str):
            self._device = torch.device(device)
        elif isinstance(device, torch.device):
            self._device = device

        return self
