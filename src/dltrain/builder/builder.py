import string
import random
from typing import Dict

import torch

from src.dltrain.forward import Forward, SimpleForward
from src.dltrain.delineator import Delineator, TrainEvalSetDelineator, RandomSplitDelineator
from src.dltrain.transform import Container
from src.dltrain.dataset import DLDataset

from torch.nn import Module, CrossEntropyLoss, MSELoss
from torch.optim import Optimizer, Adam
from torch import device
from torch.optim.lr_scheduler import LRScheduler

from src.dltrain.options import TrainOptions
from src.dltrain.evaluation import EvaluationHandler
from src.dltrain.models import create_native_model, PyTorchNativeCNN, MultilayerPerceptron
from src.dltrain.transform import Transform

from .optimizer import OptimizerBuilder
from .scheduler import SchedulerBuilder


__all__ = [
    'TaskBuilder'
]

class TaskBuilder:
    def __init__(self, task_name=None):
        self._task_name: str = task_name if task_name is not None else ''.join(
            random.choices(string.digits + string.ascii_letters, k=5))

        self._seed = 3407

        self._model: Module = None

        self._start_checkpoint: str = None
        self._save_checkpoint: bool = False

        self._optimizer_type: type[Optimizer] = Adam
        self._optimizer_parameters: dict = {}

        self._criterion: Module = None

        self._scheduler_type: type[LRScheduler] = None
        self._scheduler_parameters: dict = {}

        self._epochs: int = 10
        self._batch_size: int = 16
        self._forward: Forward = SimpleForward()
        self._delineator: Delineator = None

        self._train_evaluation_handlers: Dict[str, EvaluationHandler] = None
        self._eval_evaluation_handlers: Dict[str, EvaluationHandler] = None

        self._features_transform: Transform = None
        self._targets_transform: Transform = None

        self._device: device = torch.device('cpu')

    def add_features_transform(self, transform: Transform):
        if self._features_transform is None:
            self._features_transform = []
        self._features_transform.append(transform)

    def add_targets_transform(self, transform: Transform):
        if self._targets_transform is None:
            self._targets_transform = []
        self._targets_transform.append(transform)

    def use_task_name(self, task_name):
        self._task_name = task_name

    def use_checkpoint(self, checkpoint_path):
        self._start_checkpoint = checkpoint_path

    def use_model(self, model: Module):
        self._model = model

    def use_mlp(self, features, targets, layers=None, activation='sigmoid'):
        model = MultilayerPerceptron(features, targets, layers, activation)
        self._model = model

    def use_pytorch_model(self, model_name: str, num_classes: int, pretrained: bool = False):
        model = create_native_model(model_name, num_classes, pretrained=pretrained)
        model = PyTorchNativeCNN(model)
        self._model = model

    def use_forward(self, forward: Forward):
        self._forward = forward

    def use_train_eval_set(self, train_set: DLDataset, eval_set: DLDataset):
        self._delineator = TrainEvalSetDelineator(train_set, eval_set)

    def use_cuda(self):
        self._device = torch.device('cuda')

    def use_cpu(self):
        self._device = torch.device('cpu')

    def use_epoch(self, epochs: int):
        self._epochs = epochs

    def use_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def use_random_split_dataset(self, dataset: DLDataset, train=0.8, eval=0.2):
        self._delineator = RandomSplitDelineator(dataset, train, eval)

    def use_device(self, device):
        self._device = torch.device(device)

    def use_save_checkpoint(self, saved=False):
        self._save_checkpoint = saved

    def use_criterion(self, criterion: Module):
        self._criterion = criterion

    def use_scheduler(self, scheduler_builder: SchedulerBuilder):
        self._scheduler_type = scheduler_builder.get_type()
        self._scheduler_parameters = scheduler_builder.get_parameters()

    def use_optimizer(self, optimizer_builder: OptimizerBuilder):
        self._optimizer_type = optimizer_builder.get_type()
        self._optimizer_parameters = optimizer_builder.get_parameters()

    def add_train_evaluation_handler(self, name: str, handler: EvaluationHandler):
        if self._train_evaluation_handlers is None:
            self._train_evaluation_handlers = {}
        self._train_evaluation_handlers[name] = handler

    def add_eval_evaluation_handler(self, name: str, handler: EvaluationHandler):
        if self._eval_evaluation_handlers is None:
            self._eval_evaluation_handlers = {}
        self._eval_evaluation_handlers[name] = handler

    def use_config(self, **configs):
        for key in configs:
            if f'_{key}' in self.__dict__:
                self.__dict__[f'_{key}'] = configs[key]

    def use_mse(self):
        self._criterion = MSELoss()

    def use_cross_entry(self):
        self._criterion = CrossEntropyLoss()

    def build(self):
        options = TrainOptions()
        for key in self.__dict__:
            real_key = key[1:]
            if real_key in options.__dict__:
                options.__dict__[real_key] = self.__dict__[key]

        options.features_transform = Container(options.features_transform)
        options.targets_transform = Container(options.targets_transform)

        return options
