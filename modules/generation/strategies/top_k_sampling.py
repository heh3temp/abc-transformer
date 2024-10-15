import torch
from torch.nn import functional as F

from .ancestral_sampling import AncestralSampling


class TopKSampling(AncestralSampling):

    def __init__(self, temperature: float=1.0, k=3) -> None:
        super().__init__(temperature)

        self.k = k

    def __call__(self, logits: torch.tensor) -> torch.tensor:
        
        top_k_logits, _ = torch.topk(logits, k=self.k)
        logit_greatest_lower_bound = top_k_logits[-1]
        logits[logits < logit_greatest_lower_bound] = -float("inf")

        return super().__call__(logits)