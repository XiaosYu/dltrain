import os
from dataclasses import dataclass
from collections import OrderedDict

from .options import TrainOptions

import torch


__all__ = [
    'CheckPoint'
]

@dataclass
class CheckPoint:
    epoch: int  # Train Epoch

    optimizer_state_dict: OrderedDict
    scheduler_state_dict: OrderedDict

    total_train_loss: list
    total_eval_loss: list

    total_train_evaluation: dict
    total_eval_evaluation: dict

    options: TrainOptions

    best_eval_loss: float

    def save(self, folder):
        os.makedirs(f'{folder}/checkpoints', exist_ok=True)
        filename = os.path.join(folder, 'checkpoints', f'{self.options.task_name}_{self.epoch}.checkpoint')
        torch.save(self, filename)




