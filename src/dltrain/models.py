from torch.nn import Module, Linear, Sigmoid, ReLU, LeakyReLU, Sequential
from torchvision.models import GoogLeNet
from torchvision.models import (
    googlenet, alexnet,

    resnet18, resnet34, resnet50, resnet101, resnet152,

    vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,

    vit_b_16, vit_h_14, vit_b_32, vit_l_16, vit_l_32,

    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,

    efficientnet_v2_s, efficientnet_v2_l, efficientnet_v2_m, efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,

    densenet121, densenet161, densenet169, densenet201,

    regnet_x_8gf, regnet_x_1_6gf, regnet_y_8gf, regnet_y_400mf, regnet_y_128gf, regnet_y_1_6gf, regnet_x_3_2gf,
    regnet_x_16gf, regnet_x_32gf, regnet_x_400mf, regnet_y_800mf, regnet_x_800mf, regnet_y_3_2gf, regnet_y_16gf,
    regnet_y_32gf,

    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,

    swin_b, swin_t, swin_s, swin_v2_b, swin_v2_t, swin_v2_s,

    mnasnet0_5, mnasnet1_0, mnasnet1_3, mnasnet0_75
)

__Model__ = [
    googlenet, alexnet,

    resnet18, resnet34, resnet50, resnet101, resnet152,

    vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,

    vit_b_16, vit_h_14, vit_b_32, vit_l_16, vit_l_32,

    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,

    efficientnet_v2_s, efficientnet_v2_l, efficientnet_v2_m, efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,

    densenet121, densenet161, densenet169, densenet201,

    regnet_x_8gf, regnet_x_1_6gf, regnet_y_8gf, regnet_y_400mf, regnet_y_128gf, regnet_y_1_6gf, regnet_x_3_2gf,
    regnet_x_16gf, regnet_x_32gf, regnet_x_400mf, regnet_y_800mf, regnet_x_800mf, regnet_y_3_2gf, regnet_y_16gf,
    regnet_y_32gf,

    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,

    swin_b, swin_t, swin_s, swin_v2_b, swin_v2_t, swin_v2_s,

    mnasnet0_5, mnasnet1_0, mnasnet1_3, mnasnet0_75
]

__Activation__ = [
    Sigmoid, ReLU, LeakyReLU
]

__all__ = [
    'create_native_model',
    'PyTorchNativeCNN',
    'MultilayerPerceptron'
]


def create_native_model(model_name, num_classes, **kwargs):
    for model_constructor in __Model__:
        name = model_constructor.__name__
        if model_name == name:
            model = model_constructor(num_classes=num_classes, **kwargs)
            return model


class PyTorchNativeCNN(Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.is_google = False

        if isinstance(backbone, GoogLeNet):
            self.is_google = True

    def forward(self, data):
        if data.shape[1] != 3:
            data = data.expand(data.shape[0], 3, data.shape[2], data.shape[3])
        out = self.backbone(data)
        if not self.is_google:
            return out
        else:
            if self.training:
                return out.logits + 0.3 * out.aux_logits2 + 0.3 * out.aux_logits1
            else:
                return out


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
            modules.append(Linear(last_num, targets))

        self.layers = Sequential(
            *modules
        )

    def forward(self, data):
        data = data.reshape(data.shape[0], -1)
        return self.layers(data)
