from abc import ABC
from typing import Callable, Optional

import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.group_actions import translation
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
        kernel_accepts_tokens: bool = False,
        group_action: Callable = translation,
        phi_t: Optional[nn.Module] = None,
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
            kernel_accepts_tokens,
            group_action,
            phi_t,
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
        "x: [m, n, dx]",
        "t: [m, n, dt]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dx]",
        "return[1]: [m, n, dt]",
    )
    def attn_block(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, t = self.attn(x, t, mask=mask)
        return self.attn_dropout(x), t

    @check_shapes(
        "x: [m, n, dx]",
        "t: [m, n, dt]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dx]",
        "return[1]: [m, n, dt]",
    )
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.norm_first:
            x_update, t = self.attn_block(self.norm1(x), t, mask)
            x = x + x_update
            x = x + self.ff_block(self.norm2(x))
        else:
            x_update, t = self.attn_block(x, t, mask)
            x = x_update + self.norm1(x_update)
            x = self.norm2(x + self.ff_block(x))

        return x, t


class MultiHeadCrossTEAttentionLayer(MultiHeadTEAttentionLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, attention=MultiHeadCrossTEAttention, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nk, dt]",
        "mask: [m, nq, nk]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nq, dt]",
    )
    def attn_block(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq, tq = self.attn(xq, xk, tq, tk, mask=mask)
        return self.attn_dropout(xq), tq

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nk, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nk, dt]",
        "mask: [m, nq, nk]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nq, dt]",
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
            xq_update, tq = self.attn_block(
                self.norm1(xq), self.norm1(xk), tq, tk, mask
            )
            xq = xq + xq_update
            xq = xq + self.ff_block(self.norm2(xq))
        else:
            xq_update, tq = self.attn_block(xq, xk, tq, tk, mask)
            xq = xq + self.norm1(xq_update)
            xq = self.norm2(xq + self.ff_block(xq))

        return xq, tq
