import torch
from torch.nn import functional as F

from .strategy import Strategy


class AncestralSampling(Strategy):

    def __init__(self, temperature: float=1.0) -> None:
        super().__init__()

        self.temperature = temperature

    def __call__(self, logits: torch.tensor) -> torch.tensor:
        logits /= self.temperature
        probs = F.softmax(logits, dim=-1)
        next = torch.multinomial(probs, num_samples=1)
        
        return next
        
        
