from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .kernels import Kernel


class MultiHeadTEAttention(nn.Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        token_attention: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embed_dim)

        self.kernel = kernel
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        # Whether or not to include attention between tokens.
        self.token_attention = token_attention
        if token_attention:
            self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
            self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nkv, dt]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dx]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes multi-head translation equivariant attention.

        Args:
            xq (torch.Tensor): Query token.
            xk (torch.Tensor): Key token.
            xv (torch.Tensor): Value token.
            tq (torch.Tensor): Query inputs.
            tk (torch.Tensor): Key inputs.
            mask (Optional[torch.Tensor], optional): Query-key mask. Defaults to None.

        Returns:
            torch.Tensor: Output of attention mechanism.
        """
        # Compute translation equivariant attention.
        tq_ = tq[:, :, None, :]
        tk_ = tk[:, None, :, :]

        # Compute pairwise differences.
        # (m, nq, nkv, dx).
        diff = tq_ - tk_

        # (m, {1, h}, nq, nkv).
        dots = self.kernel(diff, mask)

        if self.token_attention:
            # Compute token attention.
            q = self.to_q(xq)
            k = self.to_k(xk)

            # Each of shape (m, num_heads, n, head_dim).
            q, k = map(
                lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
                (q, k),
            )

            # (m, h, nq, nk).
            token_dots = (q @ k.transpose(-1, -2)) * self.scale

            if mask is not None:
                mask = einops.repeat(mask, "m n p -> m h n p", h=self.num_heads)
                token_dots = torch.masked_fill(dots, mask, -float("inf"))

            dots = dots + token_dots

        attn = dots.softmax(dim=-1)

        # Multiply by values.
        v = self.to_v(xv)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        out = attn @ v
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class MultiHeadSelfTEAttention(MultiHeadTEAttention):
    @check_shapes(
        "x: [m, n, dx]", "t: [m, n, dt]", "mask: [m, n, n]", "return: [m, n, dx]"
    )
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().propagate(x, x, x, t, t, mask)


class MultiHeadCrossTEAttention(MultiHeadTEAttention):
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
    ):
        return super().propagate(xq, xk, xk, tq, tk, mask)
