from __future__ import annotations

import math
from typing import Tuple, Optional, List, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F

from .autoregressive_model import AutoregressiveModel
from modules.configs import TransformerConfig
from modules.generation.strategies import Strategy


class FeedForward(nn.Module):
    
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.embed_size, config.feed_forward_expansion * config.embed_size),
            nn.GELU(),
            nn.Linear(config.feed_forward_expansion * config.embed_size, config.embed_size),
            nn.Dropout(config.feed_forward_dropout_p)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.network(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        
        super().__init__()

        assert config.embed_size % config.num_heads == 0, "embedding size must be divisible by the number of heads"

        self.bidirectional = config.bidirectional
        self.head_size = config.embed_size // config.num_heads
        self.embed_size = config.embed_size
        self.num_heads = config.num_heads

        self.attention_dropout = nn.Dropout(config.dropout_attention_p)
        self.projection_dropout = nn.Dropout(config.dropout_projection_p)

        self.key = nn.Linear(config.embed_size, config.embed_size)
        self.query = nn.Linear(config.embed_size, config.embed_size)
        self.value = nn.Linear(config.embed_size, config.embed_size)

        self.projection = nn.Linear(config.embed_size, config.embed_size)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.context_size, config.context_size)).view(1, 1, config.context_size, config.context_size))


    def forward(self, x: torch.tensor) -> torch.tensor:
        bach_size, seq_len, embed_size = x.shape

        keys = self.key(x).reshape(bach_size, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bach_size, num_heads, seq_len, head_size)
        queries = self.query(x).reshape(bach_size, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bach_size, num_heads, seq_len, head_size)
        values = self.value(x).reshape(bach_size, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bach_size, num_heads, seq_len, head_size)

        # (bach_size, num_heads, seq_len, head_size) x (bach_size, num_heads, head_size, seq_len) -> (bach_size, num_heads, seq_len, seq_len)
        attention_weights = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        if not self.bidirectional:
            attention_weights = attention_weights.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        out = attention_weights @ values # (bach_size, num_heads, seq_len, seq_len) x (bach_size, num_heads, seq_len, head_size) -> (bach_size, num_heads, seq_len, head_size)
        out = out.transpose(1, 2).reshape(bach_size, seq_len, embed_size) # (bach_size, seq_len, embed_size)

        out = self.projection_dropout(self.projection(out))

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.feed_forward_block = nn.Sequential(
            nn.LayerNorm(config.embed_size),
            FeedForward(config)
        )
        self.attention_block = nn.Sequential(
            nn.LayerNorm(config.embed_size),
            MultiHeadedSelfAttention(config)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.attention_block(x)
        x = x + self.feed_forward_block(x)

        return x


class Transformer(AutoregressiveModel):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.context_size = config.context_size
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
        self.feature_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.layer_norm = nn.LayerNorm(config.embed_size)
        self.dropout = nn.Dropout(config.embed_dropout_p)
        self.out_projection = nn.Linear(config.embed_size, config.vocab_size)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(F"Number of parameters: {n_params/1e6}M")

    def _init_weights(self, module: nn.Module) -> None:
       
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.feature_embedding(x) + self.position_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.layer_norm(x)
        x = self.out_projection(x)

        return x
    
    def training_step(self, x: torch.tensor, y: torch.tensor) -> float:

        self.optimizer.zero_grad()

        logits = self.forward(x)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), y.view(-1))
        
        loss.backward()
        
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip)

        self.optimizer.step()

        return loss.item()
    
    def configure_optimizers(
            self,
            weight_decay: Optional[float]=0.0,
            learning_rate: Optional[float]=0.001,
            betas: Optional[Tuple[float, float]]=(0.9, 0.999),
            grad_norm_clip: Optional[float]=None
        ) -> None:

        optim_groups = self._determine_optim_groups(weight_decay)
        self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        self.grad_norm_clip = grad_norm_clip

    def _determine_optim_groups(self, weight_decay: float) -> List[Dict[str, Union[List[str], float]]]:
        weight_decay_params = set()
        no_weight_decay_params = set()

        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters():
                full_parameter_name = f"{module_name}.{parameter_name}" if module_name else parameter_name

                if parameter_name.endswith('bias'):

                    no_weight_decay_params.add(full_parameter_name)
                elif parameter_name.endswith('weight') and isinstance(module, whitelist_weight_modules):

                    weight_decay_params.add(full_parameter_name)
                elif parameter_name.endswith('weight') and isinstance(module, blacklist_weight_modules):

                    no_weight_decay_params.add(full_parameter_name)

        all_params = {parameter_name: parameter for parameter_name, parameter in self.named_parameters()}
        
        intersection_params = weight_decay_params & no_weight_decay_params
        union_params = weight_decay_params | no_weight_decay_params
        not_considered = all_params.keys() - union_params

        assert len(intersection_params) == 0, f"Could not deremine whether parameters {intersection_params} should experience weight decay"
        assert len(not_considered) == 0, f"Parameters {not_considered} were not not taken into account when determinig whether parameters should experience weight decay" \

        optim_groups = [
            {"params": [all_params[parameter_name] for parameter_name in sorted(list(weight_decay_params))], "weight_decay": weight_decay},
            {"params": [all_params[parameter_name] for parameter_name in sorted(list(no_weight_decay_params))], "weight_decay": 0.0},
        ]

        return optim_groups

    @torch.no_grad()
    def generate(
        self,
        sequence: torch.tensor,
        max_len: int,
        EOS: int,
        strategy: Strategy
    ) -> torch.tensor:

        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)
        
        self.eval()

        reached_EOS = False
        i = 0
        while i <= max_len and not reached_EOS:

            context = sequence if sequence.shape[1] <= self.context_size else sequence[:, -self.context_size:]
            logits = self.forward(context)[:, -1, :]
            predicted_token = strategy(logits.squeeze())

            if predicted_token.item() != EOS:
                sequence = torch.cat((sequence, predicted_token.unsqueeze(0)), dim=1)
            else:
                reached_EOS = True
            
            i += 1

        return sequence.squeeze()

