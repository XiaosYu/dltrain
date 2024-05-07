from .core import Wizard
from ..models import MultilayerPerceptron, PyTorchNativeCNN, create_native_model, Module


class ModelWizard(Wizard):
    def __init__(self):
        self._model = None

    def use_model(self, model: Module):
        self._model = model
        return self

    def use_mlp(self, features, targets, layers=None, activation='sigmoid'):
        model = MultilayerPerceptron(features, targets, layers, activation)
        self._model = model

    def use_pytorch_model(self, model_name: str, num_classes: int, pretrained: bool = False):
        model = create_native_model(model_name, num_classes, pretrained=pretrained)
        model = PyTorchNativeCNN(model)
        self._model = model

    def use_resnet18(self, num_classes):
        return self.use_pytorch_model('resnet18', num_classes)
