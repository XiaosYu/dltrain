from typing import List, Dict

from .forward import Forward
from .delineator import Delineator
from .evaluation import EvaluationHandler
from .transform import Transform
from torch.nn import Module
from torch.optim import Optimizer
from torch import device
from torch.optim.lr_scheduler import LRScheduler


__all__ =[
    'TrainOptions'
]

class TrainOptions:
    def __init__(self):
        self.task_name: str = None
        self.seed: int = None

        self.start_checkpoint: str = None
        self.save_checkpoint: bool = True

        self.model: Module = None

        self.optimizer_type: type[Optimizer] = None
        self.optimizer_parameters: dict = {}

        self.criterion: Module = None

        self.scheduler_type: type[LRScheduler] = None
        self.scheduler_parameters: dict = {}

        self.epochs: int = None
        self.batch_size: int = None
        self.forward: Forward = None
        self.delineator: Delineator = None

        self.train_evaluation_handlers: Dict[str, EvaluationHandler] = None
        self.eval_evaluation_handlers: Dict[str, EvaluationHandler] = None

        self.features_transform: Transform = None
        self.targets_transform: Transform = None

        self.device: device = None
