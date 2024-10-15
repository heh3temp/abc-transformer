import torch
from typing import Iterable, List, Tuple, Dict

from modules.configs import CharTokenizerConfig


class CharTokenizer:

    def __init__(self, config: CharTokenizerConfig) -> None:

        self.config = config
        self._id_to_char, self._char_to_id = self._build_lookups(config.vocab)

    @classmethod
    def from_corpus(cls, corpus: List[str], sos, eos):
        vocab = CharTokenizer._build_vocab(corpus)
        config = CharTokenizerConfig(eos, sos, vocab)

        return cls(config)
    
    @property
    def SOS(self) -> int:
        return self.encode([self.config.SOS]).item()

    @property
    def EOS(self) -> int:
        return self.encode([self.config.EOS]).item()
    
    @property
    def vocab(self) -> List[str]:
        return self._id_to_char

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_char)
    
    @staticmethod
    def _build_vocab(corpus: List[str]) -> List[str]:
        vocab = sorted(set(corpus))
        return vocab

    def _build_lookups(self, vocab: List[str]) -> Tuple[List[str], Dict[str, int]]:
        id_to_char = vocab
        char_to_id = {char: id for id, char in enumerate(id_to_char)}

        return id_to_char, char_to_id
    
    def encode(self, text: str) -> torch.tensor:
        encoded = torch.zeros(len(text), dtype=torch.long)
        for i, char in enumerate(text):
            encoded[i] = self._char_to_id[char]

        return encoded
    
    def decode(self, data: Iterable) -> str:
        decoded = ""

        for i in range(len(data)):
            decoded += self._id_to_char[data[i]]

        return decoded
   