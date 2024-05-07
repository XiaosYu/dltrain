import string
import random


from ..transform import Container
from .inject import InjectWizard
from ..options import TrainOptions
from .optimizer import OptimizerWizard
from .scheduler import SchedulerWizard
from .base import BaseWizard
from .criterion import CriterionWizard
from .evaluation import EvaluationWizard
from .model import ModelWizard
from .transforms import TransformWizard
from .dataset import DatasetWizard
from .delineator import DelineatorWizard
from .forward import ForwardWizard

__all__ = [
    'TaskBuilder'
]


class TaskBuilder:
    def __init__(self, task_name=None):
        self.task_name: str = task_name if task_name is not None else ''.join(
            random.choices(string.digits + string.ascii_letters, k=5))

        self.forward = ForwardWizard()
        self.evaluation_handler = EvaluationWizard()
        self.optimizer = OptimizerWizard()
        self.base = BaseWizard()
        self.criterion = CriterionWizard()
        self.scheduler = SchedulerWizard()
        self.model = ModelWizard()
        self.transform = TransformWizard()
        self.dataset = DatasetWizard()
        self.delineator = DelineatorWizard()
        self.inject = InjectWizard()

    def build(self):
        options = TrainOptions()
        options.task_name = self.task_name

        # 载入优化器
        optimizer = self.optimizer.get_kwargs()
        if optimizer['type'] is not None and optimizer['parameters'] is not None:
            options.optimizer_type = optimizer['type']
            options.optimizer_parameters = optimizer['parameters']

        # 载入基本信息
        for key, value in self.base.get_kwargs().items():
            if key in options.__dict__:
                options.__dict__[key] = value

        # 载入损失函数
        options.criterion = self.criterion.get_kwargs()['criterion']

        # 载入验证手段
        handlers = self.evaluation_handler.get_kwargs()
        options.train_evaluation_handlers = handlers['train_evaluation_handlers']
        options.eval_evaluation_handlers = handlers['eval_evaluation_handlers']

        # 载入策略器
        scheduler = self.scheduler.get_kwargs()
        if scheduler['type'] is not None and scheduler['parameters'] is not None:
            options.scheduler_type = scheduler['type']
            options.scheduler_parameters = scheduler['parameters']

        # 载入模型
        model = self.model.get_kwargs()['model']
        options.model = model

        # 载入数据变换
        transform = self.transform.get_kwargs()
        options.features_transform = Container(transform['features_transform'])
        options.targets_transform = Container(transform['targets_transform'])

        # 数据集设置
        options.delineator = self.delineator.get_kwargs()['delineator']

        # 设置前馈
        options.forward = self.forward.get_kwargs()['forward']

        # 载入注入
        injects = self.inject.get_kwargs()['injects']
        if hasattr(options.forward, 'bind'):
            for inject in injects:
                options.forward.bind(inject)

        return options
