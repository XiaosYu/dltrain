from .component import ConvClassificationHeader, LinearClassificationHeader
from .wrapper import create_native_model
from .model import TaskModel
from abc import ABCMeta
from torch import nn


class ModelBuilder(metaclass=ABCMeta):
    def __init__(self):
        self._body = None
        self._header = None

    def build(self):
        return TaskModel(self._body, self._header)


class ImageClassificationBuilder(ModelBuilder):
    def __init__(self, num_classes, input_size):
        super().__init__()

        if len(input_size) != 3:
            raise ValueError('Input size must be like [channels, width, height]')
        self.input_size = input_size

        body = nn.Sequential()

        channels = input_size[0]
        if channels != 3:
            body.append(nn.Conv2d(channels, 3, 2, 1, 0))

        self._header = LinearClassificationHeader(num_classes)
        backbone = create_native_model('mobilenet_v3_small', num_classes=10)
        self._body = backbone.features


