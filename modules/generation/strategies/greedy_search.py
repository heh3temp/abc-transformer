import torch
from torch.nn import functional as F

from .strategy import Strategy


class GreedySearch(Strategy):

    def __call__(self, logits: torch.tensor) -> torch.tensor:

        next = torch.argmax(logits, keepdim=True)
        
        return next