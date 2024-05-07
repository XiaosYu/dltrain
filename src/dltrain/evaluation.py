import random
import string
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor

__all__ = [
    'Evaluation',
    'EvaluationHandler',
    'Accuracy',
    'ConfusionMatrix',
    'RootMeanSquareError',
    'RSquare',
    'MeanSquareError',
    'SaveImage'
]


class Evaluation:
    def __init__(self):
        self.predictions = None
        self.exacts = None

    def append(self, prediction):
        self.predictions = prediction if self.predictions is None else torch.cat([self.predictions, prediction], dim=0)

    def reset(self):
        self.predictions = None


class EvaluationHandler(metaclass=ABCMeta):
    def __init__(self, drawable=False):
        self.drawable = drawable

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, epoch_predictions: Tensor, epoch_exacts: Tensor):
        pass


class Accuracy(EvaluationHandler):
    def __init__(self, drawable=False):
        super().__init__(drawable)

    def compute(self, epoch_predictions, epoch_exacts):
        return int(torch.sum(epoch_predictions.argmax(dim=1) == epoch_exacts).item()) / epoch_exacts.shape[0]


class ConfusionMatrix(EvaluationHandler):
    def __init__(self, drawable=False):
        super().__init__(drawable)
        from sklearn.metrics import confusion_matrix
        self.func = confusion_matrix

    def compute(self, epoch_predictions, epoch_exacts):
        matrix = self.func(epoch_exacts.cpu().detach().numpy(), epoch_predictions.argmax(dim=1).cpu().detach().numpy())
        return matrix


class RootMeanSquareError(EvaluationHandler):
    def __init__(self, drawable=False):
        super().__init__(drawable)

    def compute(self, epoch_predictions: Tensor, epoch_exacts: Tensor):
        mse = torch.nn.functional.mse_loss(epoch_predictions, epoch_exacts)
        rmse = torch.sqrt(mse)
        return float(rmse)


class MeanSquareError(EvaluationHandler):
    def __init__(self, drawable=False):
        super().__init__(drawable)

    def compute(self, epoch_predictions: Tensor, epoch_exacts: Tensor):
        mse = torch.nn.functional.mse_loss(epoch_predictions, epoch_exacts)
        return float(mse)


class RSquare(EvaluationHandler):
    def __init__(self, drawable=False):
        super().__init__(drawable)
        from sklearn.metrics import r2_score
        self.func = r2_score

    def compute(self, epoch_predictions: Tensor, epoch_exacts: Tensor):
        r2 = self.func(epoch_exacts.cpu().detach().numpy(), epoch_predictions.cpu().detach().numpy())
        return float(r2)


class SaveImage(EvaluationHandler):
    def __init__(self, folder: str, name_generator=None, file_format='png', save_function=None):
        super().__init__(False)

        if name_generator is not None:
            self.name_generator = name_generator
        else:
            self.name_generator = lambda: random.choices(string.ascii_letters, k=10)

        self.file_format = file_format
        self.folder = folder

        if save_function is not None:
            self.save_function = save_function
        else:
            from torchvision.transforms.functional import to_pil_image
            self.save_function = to_pil_image

    def compute(self, epoch_predictions: Tensor, epoch_exacts: Tensor):
        if len(epoch_predictions.shape) == 4:
            for image_tensor in epoch_predictions:
                image_path = f'{self.folder}/{self.name_generator()}.{self.file_format}'
                image = self.save_function(image_tensor)
                image.save(image_path)

        elif len(epoch_predictions.shape) == 3:
            image_path = f'{self.folder}/{self.name_generator()}.{self.file_format}'
            image = self.save_function(epoch_predictions)
            image.save(image_path)
