from .base import Wizard
from ..forward import SimpleForward, Forward, InjectForward


class ForwardWizard(Wizard):
    def __init__(self):
        self._forward = SimpleForward()

    def use_forward(self, forward: Forward):
        self._forward = forward
        return self

    def use_simple(self):
        return self.use_forward(SimpleForward())

    def use_inject(self):
        inject = InjectForward()
        self.use_forward(inject)
        return inject
