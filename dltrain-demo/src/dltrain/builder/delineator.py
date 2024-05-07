from abc import ABCMeta, abstractmethod

from .base import Wizard
from ..delineator import RandomSplitDelineator, TrainEvalSetDelineator, Delineator, DLDataset


class DelineatorWizard(Wizard):
    def __init__(self):
        self._delineator = None

    def use_delineator(self, delineator: Delineator):
        self._delineator = delineator
        return self

    def use_random_split(self, dataset: DLDataset, train_rate=0.8, eval_rate=0.2):
        return self.use_delineator(RandomSplitDelineator(dataset, train_rate, eval_rate))

    def use_train_eval(self, train_set: DLDataset, eval_set: DLDataset = None):
        return self.use_delineator(TrainEvalSetDelineator(train_set, eval_set))


