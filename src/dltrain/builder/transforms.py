from ..transform import Transform, Standardize, Resize
from .core import Wizard


class TransformWizard(Wizard):
    def __init__(self):
        self._features_transform = []
        self._targets_transform = []

    def add_transform(self, transform: Transform, is_feature=True):
        if is_feature:
            self._features_transform.append(transform)
        else:
            self._targets_transform.append(transform)

        return self

    def add_resize(self, size, is_feature=True):
        self.add_transform(Resize(size), is_feature)
        return self

    def add_standardize(self, is_feature=True):
        self.add_transform(Standardize(), is_feature)
        return self
