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
        num_heads: int,
        head_dim: int,
        attention: MultiHeadTEAttention,
        kernel: Kernel,
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        token_attention: bool = True,
        token_kernel: bool = False,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(
            kernel,
            embed_dim,
            num_heads,
            head_dim,
            p_dropout,
            token_attention,
            token_kernel,
        )

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
        "x: [m, n, dx]", "t: [m, n, dt]", "mask: [m, n, n]", "return: [m, n, dx]"
    )
    def attn_block(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, t, mask=mask)
        return self.attn_dropout(x)

    @check_shapes(
        "x: [m, n, dx]", "t: [m, n, dt]", "mask: [m, n, n]", "return: [m, n, dx]"
    )
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn_block(self.norm1(x), t, mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = x + self.norm1(x + self.attn_block(x, t, mask))
            x = self.norm2(x + self.ff_block(x))

        return x


class MultiHeadCrossTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attention=MultiHeadCrossTEAttention, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nk, dt]",
        "mask: [m, nq, nk]",
        "return: [m, nq, dx]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq = self.attn(xq, xk, tq, tk, mask=mask)
        return self.attn_dropout(xq)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nk, dt]",
        "mask: [m, nq, nk]",
        "return: [m, nq, dx]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_first:
            xq = xq + self.attn_block(self.norm1(xq), self.norm1(xk), tq, tk, mask)
            xq = xq + self.ff_block(self.norm2(xq))
        else:
            xq = xq + self.norm1(xq + self.attn_block(xq, xk, tq, tk, mask))
            xq = self.norm2(xq + self.ff_block(xq))

        return xq
