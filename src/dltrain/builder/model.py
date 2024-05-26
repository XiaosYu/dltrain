from .core import Wizard
from ..models import MultilayerPerceptron, PyTorchNativeCNN, create_native_model, ImageClassificationBuilder
from torch.nn import Module


class ModelWizard(Wizard):
    def __init__(self):
        self._model = None

    def __getattribute__(self, name):
        if name == '_model':
            if isinstance(self._model, Module):
                return self._model
            else:
                self._model = self._model.build()
                return self._model

        else:
            return object.__getattribute__(self, name)

    def use_image_classification_builder(self, num_classes, channels, width, height) -> ImageClassificationBuilder:
        self._model = ImageClassificationBuilder(num_classes=10, input_size=(channels, width, height))
        return self._model

    def use_model(self, model: Module):
        self._model = model

    def use_mlp(self, features, targets, layers=None, activation='sigmoid'):
        model = MultilayerPerceptron(features, targets, layers, activation)
        self._model = model

    def use_pytorch_model(self, model_name: str, num_classes: int, pretrained: bool = False):
        model = create_native_model(model_name, num_classes, pretrained=pretrained)
        model = PyTorchNativeCNN(model)
        self._model = model

    def use_resnet18(self, num_classes):
        return self.use_pytorch_model('resnet18', num_classes)
