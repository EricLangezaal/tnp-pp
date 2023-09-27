import torch
from torch import nn

from .attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        feedforward_dim: int,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
