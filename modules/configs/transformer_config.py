from dataclasses import dataclass

from .config import Config


@dataclass
class TransformerConfig(Config):
    num_blocks: int
    embed_size: int
    vocab_size: int
    context_size: int
    feed_forward_expansion: int
    num_heads: int
    dropout_attention_p: float
    dropout_projection_p: float
    feed_forward_dropout_p: float
    embed_dropout_p: float
    bidirectional: bool