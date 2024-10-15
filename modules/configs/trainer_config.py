from dataclasses import dataclass

from .config import Config


@dataclass
class TrainerConfig(Config):
    epochs: int
    weight_decay: float
    learning_rate: float
    betas: tuple
    grad_norm_clip: float
    batch_size: int
    num_workers: int