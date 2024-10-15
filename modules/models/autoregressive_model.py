from abc import ABC, abstractmethod

from .model import Model


class AutoregressiveModel(Model, ABC):

    @abstractmethod
    def generate(self):
        pass