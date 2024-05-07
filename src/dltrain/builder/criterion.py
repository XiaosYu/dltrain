from .core import Wizard

from torch import nn

class CriterionWizard(Wizard):
    def __init__(self):
        self._criterion = None

    def use_criterion(self, criterion):
        self._criterion = criterion
        return self

    def use_mse(self):
        self._criterion = nn.MSELoss()
        return self

    def use_l1(self):
        self._criterion = nn.L1Loss()
        return self

    def use_cross_entropy(self):
        self._criterion = nn.CrossEntropyLoss()
        return self

    def use_bce_with_logits(self):
        self._criterion = nn.BCEWithLogitsLoss()
        return self

    def use_bce(self):
        self._criterion = nn.BCELoss()
        return self