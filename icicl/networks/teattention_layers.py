from abc import ABC
from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn

from .kernels import Kernel
from .teattention import (
    MultiHeadCrossTEAttention,
    MultiHeadSelfTEAttention,
    MultiHeadTEAttention,
)


class MultiHeadTEAttentionLayer(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        attention: MultiHeadTEAttention,
        kernel: Kernel,
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(kernel, embed_dim, head_dim, p_dropout)

        # Feedforward model.
        self.ff_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(p_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_first = norm_first

        self.attn_dropout = nn.Dropout(p_dropout)


class MultiHeadSelfTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attention=MultiHeadSelfTEAttention, **kwargs)

    @check_shapes(
        "x: [m, n, d]", "y: [m, n, dy]", "mask: [m, n, n]", "return: [m, n, dy]"
    )
    def attn_block(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.attn(x, y, mask=mask)
        return self.attn_dropout(y)

    @check_shapes(
        "x: [m, n, d]", "y: [m, n, dy]", "mask: [m, n, n]", "return: [m, n, dy]"
    )
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            y = y + self.attn_block(x, self.norm1(y), mask)
            y = y + self.ff_block(self.norm2(y))
        else:
            y = y + self.norm1(y + self.attn_block(x, y, mask))
            y = self.norm2(y + self.ff_block(y))

        return y


class MultiHeadCrossTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attention=MultiHeadCrossTEAttention, **kwargs)

    @check_shapes(
        "xq: [m, nq, d]",
        "xk: [m, nkv, d]",
        "yv: [m, nkv, dy]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dy]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        yv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        yv = self.attn(xq, xk, yv, mask=mask)
        return self.attn_dropout(yv)

    @check_shapes(
        "xq: [m, nq, d]",
        "xk: [m, nkv, d]",
        "yq: [m, nq, dy]",
        "yv: [m, nkv, dy]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dy]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        yq: torch.Tensor,
        yv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            yq = yq + self.attn_block(xq, xk, self.norm1(yv), mask)
            yq = yq + self.ff_block(self.norm2(yq))
        else:
            yq = yq + self.norm1(yq + self.attn_block(xq, xk, yv, mask))
            yq = self.norm2(yq + self.ff_block(yq))

        return yq
