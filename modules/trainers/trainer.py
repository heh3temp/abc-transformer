from __future__ import annotations

from typing import List, Optional

import torch
from tqdm import tqdm

from modules.callbacks import Callback
from modules.models import Model
from modules.configs import TrainerConfig


class Trainer:
    def __init__(
            self,
            config: TrainerConfig,
            device: torch.device,
            callbacks: Optional[List[Callback]]=None
    ) -> None:
        
        self.config = config
        self._device = device
        self._callbacks = callbacks if callbacks is not None else []

    def add_callback(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def fit(
            self,
            model: Model,
            train_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        
        model = model.to(self._device)
        model.train()
        model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, self.config.betas, self.config.grad_norm_clip)

        for callback in self._callbacks:
            callback.on_training_start(model)

        step = 0
        for epoch in range(1, self.config.epochs+1):
            avg_loss = 0
            print(f"Processing epoch [{epoch}/{self.config.epochs}]")
            for inputs, ground_truths in tqdm(train_dataloader):

                inputs = inputs.to(self._device)
                ground_truths = ground_truths.to(self._device)
                loss = model.training_step(inputs, ground_truths)

                avg_loss += loss / ground_truths.shape[0]

                for callback in self._callbacks:
                    callback.on_batch_end(model, epoch, loss, step)
                    
                step += 1

            avg_loss /= len(train_dataloader)
            print(f"Average loss = {avg_loss}")

            for callback in self._callbacks:
                callback.on_epoch_end(model, epoch, avg_loss)
        
        for callback in self._callbacks:
            callback.on_training_end(model, epoch, avg_loss)
