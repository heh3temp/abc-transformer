from typing import Tuple
import torch

from .train_dataset import TrainDataset

class TransformerTrainDataset(TrainDataset):
    
    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        chunk = self.data[idx:idx + self.context_size + 1]
        x = chunk[:-1].clone().detach()
        y = chunk[1:].clone().detach()

        return x, y