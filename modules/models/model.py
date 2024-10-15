from abc import abstractmethod, ABC
import torch


class Model(torch.nn.Module, ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def load_checkpoint(self, path: str, device: torch.device) -> None:
        self.configure_optimizers()

        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    @abstractmethod
    def configure_optimizers(self):
        pass
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def training_step(self):
        pass