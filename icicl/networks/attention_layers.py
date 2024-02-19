from abc import ABC
from functools import partial
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .attention import (
    BaseMultiHeadAttention,
    MultiHeadAttention,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)


class BaseMultiHeadAttentionLayer(nn.Module, ABC):
    def __init__(
        self,
        embed_dim: int,
        attention: Union[BaseMultiHeadAttention, partial[BaseMultiHeadAttention]],
        feedforward_dim: Optional[int] = None,
        p_dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_first: bool = False,
        **kwargs,
    ):
        super().__init__()
        feedforward_dim = embed_dim if feedforward_dim is None else feedforward_dim

        self.embed_dim = embed_dim
        self.attn = attention(**kwargs)

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


class MultiHeadAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        **kwargs,
    ):
        attention = partial(MultiHeadAttention, qk_dim=qk_dim, v_dim=v_dim)
        super().__init__(embed_dim=v_dim, attention=attention, **kwargs)

    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(xq, xk, xv, mask=mask)
        return self.attn_dropout(x)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # An MHA block is just the MHA operation.
        xq = self.attn_block(xq, xk, xv, mask)

        return xq


class MultiHeadSelfAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadSelfAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def attn_block(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, mask=mask)
        return self.attn_dropout(x)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn_block(self.norm1(x), mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.attn_block(x, mask))
            x = self.norm2(x + self.ff_block(x))

        return x


class MultiHeadCrossAttentionLayer(BaseMultiHeadAttentionLayer):
    def __init__(self, *, embed_dim: int, **kwargs):
        attention = partial(MultiHeadCrossAttention, embed_dim=embed_dim)
        super().__init__(embed_dim=embed_dim, attention=attention, **kwargs)

    @check_shapes(
        "xq: [m, nq, d]", "xkv: [m, nkv, d]", "mask: [m, n, n]", "return: [m, n, d]"
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xkv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(xq, xkv, mask=mask)
        return self.attn_dropout(x)

    @check_shapes(
        "xq: [m, nq, d]", "xkv: [m, nkv, d]", "mask: [m, n, n]", "return: [m, n, d]"
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            xq = xq + self.attn_block(self.norm1(xq), self.norm1(xkv), mask)
            xq = xq + self.ff_block(self.norm2(xq))
        else:
            xq = xq + self.norm1(xq + self.attn_block(xq, xkv, mask))
            xq = self.norm2(xq + self.ff_block(xq))

        return xq
