from abc import ABC, abstractmethod
import wandb

from modules.models.model import Model
from .callback import Callback


class WandbLoggingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_training_start(self, model: Model) -> None:
        wandb.watch(models=model, criterion=model.criterion, log="all", log_freq=10)

    def on_batch_end(self, model: Model, epoch: int, loss: float, step: int) -> None:
        wandb.log({"epoch": epoch, "loss": loss}, step=step)
