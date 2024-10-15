from abc import abstractmethod, ABC
import torch


class Strategy(ABC):

    @abstractmethod
    def __call__(logits: torch.tensor) -> torch.tensor:
        pass