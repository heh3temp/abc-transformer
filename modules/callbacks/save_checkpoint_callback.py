import torch

from modules.models.model import Model
from .callback import Callback


class SaveCheckpointCallback(Callback):
    def __init__(self, ckpt_path: str, period_epochs: int=1) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        self.period_epochs = period_epochs

    def _save_checkpoint(self, model: Model, filepath: str):
        state = {
                "state_dict": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
            }
        
        torch.save(state, filepath)

    def on_epoch_end(self, model: Model, epoch: int, loss: float) -> None:
        if epoch % self.period_epochs == 0:
            filepath = self.ckpt_path + type(model).__name__ + f"(epoch={epoch}).pt"
            self._save_checkpoint(model, filepath)
            
    def on_training_end(self, model: Model, epoch: int, loss: float):
        filepath = self.ckpt_path + "final.pt"
        self._save_checkpoint(model, filepath)