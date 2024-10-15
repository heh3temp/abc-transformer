from typing import List
from dataclasses import dataclass

from .config import Config


@dataclass
class CharTokenizerConfig(Config):
    EOS: str
    SOS: str
    vocab: List[str]