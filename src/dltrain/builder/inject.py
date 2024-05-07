from ..inject import GradientAcquisition
from .core import Wizard

class InjectWizard(Wizard):
    def __init__(self):
        self._injects = []

    def use_gradient_acquisition(self, folder):
        self._injects.append(GradientAcquisition(folder))
        return self