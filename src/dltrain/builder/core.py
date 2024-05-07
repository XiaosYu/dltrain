from abc import ABCMeta, abstractmethod

class Wizard(metaclass=ABCMeta):
    def get_kwargs(self):
        kwargs = {key.lstrip('_'): value for (key, value) in self.__dict__.items()}
        return kwargs
