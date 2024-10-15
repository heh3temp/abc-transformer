from abc import abstractmethod
from typing import Tuple

import torch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.tensor, context_size: int) -> None:
        self.data = data
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.data) - self.context_size
    
    @abstractmethod
    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        pass