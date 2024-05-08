from typing import Union
from torch import nn


class StandardConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, data):
        return self.conv(data)


class LinearClassificationHeader(nn.Module):
    def __init__(self, classes: int, features: Union[int, str] = 'auto'):
        super().__init__()

        if isinstance(features, int):
            features_num = features
        else:
            features_num = classes * 16

        self.avg = nn.AdaptiveAvgPool1d(output_size=features_num)
        self.header = nn.Linear(features_num, classes)

    def forward(self, data):
        data = data.reshape(data.shape[0], -1)
        data = self.avg(data)
        data = self.header(data)
        return data


class ConvClassificationHeader(nn.Module):
    def __init__(self, classes: int):
        super().__init__()

        self.linear_avg = nn.AdaptiveAvgPool1d(output_size=classes)

        self.conv_list = nn.Sequential(
            nn.Conv1d(1, 3, int(classes / 2), 2),
            nn.Conv1d(3, 1, int(classes / 2), 2)
        )

    def forward(self, data):
        data = data.reshape(data.shape[0], 1, -1)
        data = self.conv_list(data)
        data = data.reshape(data.shape[0], -1)
        data = self.linear_avg(data)
        return data


class TaskModel(nn.Module):
    def __init__(self, feature_model: nn.Module, header_model: nn.Module):
        super().__init__()

        self.feature_model = feature_model
        self.header_model = header_model

    def forward(self, data):
        data = self.feature_model(data)
        data = self.header_model(data)
        return data
