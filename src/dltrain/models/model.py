from torch.nn import (Module, Linear, Sigmoid, ReLU, LeakyReLU, Sequential, Tanh, Hardswish, Threshold, Dropout,
                      BatchNorm2d, BatchNorm1d, LazyBatchNorm1d)

__Activation__ = [
    Sigmoid, ReLU, LeakyReLU, Tanh, Hardswish,
    BatchNorm1d, BatchNorm2d, LazyBatchNorm1d
]

__all__ = [
    'MultilayerPerceptron',
    'TaskModel'
]


class MultilayerPerceptron(Module):
    def __init__(self, features, targets, layers=None, activation='sigmoid'):
        super().__init__()

        constructor = None
        for activation_constructor in __Activation__:
            name = activation_constructor.__name__
            if activation == name.lower():
                constructor = activation_constructor

        if constructor is None:
            constructor = Sigmoid

        modules = []
        if layers is None:
            hidden = int((features + targets) / 2)
            modules.append(Linear(features, hidden))
            modules.append(constructor())
            modules.append(Linear(hidden, hidden))
            modules.append(constructor())
            modules.append(Linear(hidden, targets))
        else:
            last_num = features
            for layer in layers:
                modules.append(Linear(last_num, layer))
                modules.append(constructor())
                last_num = layer
            modules.append(Linear(last_num, targets))

        self.layers = Sequential(
            *modules
        )

    def forward(self, data):
        data = data.reshape(data.shape[0], -1)
        return self.layers(data)


class TaskModel(Module):
    def __init__(self, feature_model: Module, header_model: Module):
        super().__init__()

        self.feature_model = feature_model
        self.header_model = header_model

    def forward(self, data):
        data = self.feature_model(data)
        data = self.header_model(data)
        return data
