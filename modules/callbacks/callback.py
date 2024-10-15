from abc import ABC, abstractmethod

from modules.models.model import Model


class Callback(ABC):
    
    def on_batch_end(self, model: Model, epoch: int, loss: float, step: int):
        pass

    def on_epoch_end(self, model: Model, epoch: int, loss: float):
        pass

    def on_training_start(self, model: Model):
        pass

    def on_training_end(self, model: Model, epoch: int, loss: float):
        pass