from .core import Wizard
from ..evaluation import EvaluationHandler, Accuracy, MeanSquareError, RSquare, RootMeanSquareError, ConfusionMatrix, \
    SaveImage


class EvaluationWizard(Wizard):
    def __init__(self):
        self._train_evaluation_handlers: dict[str, EvaluationHandler] = {}
        self._eval_evaluation_handlers: dict[str, EvaluationHandler] = {}

    def add_evaluation_handler(self, name: str, handler: EvaluationHandler, role: str):
        if role.lower() == 'train':
            self._train_evaluation_handlers[name] = handler
        elif role.lower() == 'eval':
            self._eval_evaluation_handlers[name] = handler

        return self

    def add_evaluation_handler_type(self, name: str, handler: type[EvaluationHandler], role: str = 'total'):
        if role.lower() == 'total':
            self._train_evaluation_handlers[name] = handler()
            self._eval_evaluation_handlers[name] = handler()
        else:
            self.add_evaluation_handler(name, handler(), role)

        return self

    def add_accuracy(self, role: str = 'total', drawable=True):
        return self.add_evaluation_handler_type('Accuracy', lambda: Accuracy(drawable), role)

    def add_confusion_matrix(self, role: str = 'total', drawable=True):
        return self.add_evaluation_handler_type('ConfusionMatrix', lambda: ConfusionMatrix(drawable), role)

    def add_save_image(self, folder: str, role: str = 'total'):
        return self.add_evaluation_handler('SaveImage', SaveImage(folder), role)
