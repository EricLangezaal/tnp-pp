"""https://github.com/rishikksh20/CrossViT-pytorch/blob/master/module.py"""
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .kernels import Kernel


class BaseMultiHeadAttention(nn.Module, ABC):
    def __init__(
        self,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        kernel: Optional[Kernel] = None,
        add_diagonal_attention: bool = False,
    ):
        super().__init__()

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == v_dim)

        self.to_q = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(qk_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, v_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

        # Optional kernel nonlinearity to apply to inner products.
        self.kernel = kernel

        # Whether to do diagonals in mhca attention.
        self.add_diagonal_attention = add_diagonal_attention

    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
        "xv: [m, nkv, dv]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dv]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.to_q(xq)
        k = self.to_k(xk)
        v = self.to_v(xv)

        # Each of shape (m, num_heads, n, head_dim).
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v),
        )

        # (m, num_heads, nq, nk).
        dots = (q @ k.transpose(-1, -2)) * self.scale

        if self.add_diagonal_attention:
            assert (
                xq.shape[-1] == xv.shape[-1]
            ), "xq and xv must have same embedding dimension."
            xq_k = self.to_k(xq)
            xq_v = self.to_v(xq)
            xq_k, xq_v = map(
                lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
                (xq_k, xq_v),
            )
            # Add diagonal self attention amongst xq.
            diag_dots = (q * xq_k).sum(-1, keepdim=True) * self.scale

            # (m, h, nq, nk + 1).
            dots = torch.cat((dots, diag_dots), dim=-1)

            # (m, h, nv + 1, head_dim).
            # v = torch.cat((v, xq_v), dim=-2)

        if self.kernel is not None:
            dots = einops.rearrange(dots, "m h nq nk -> m nq nk h")
            dots = self.kernel(dots)
            dots = einops.rearrange(dots, "m nq nk h -> m h nq nk")

        if mask is not None:
            mask = einops.repeat(mask, "m n p -> m h n p", h=self.num_heads)
            dots = torch.masked_fill(dots, mask, -float("inf"))

        attn = dots.softmax(dim=-1)

        if self.add_diagonal_attention:
            out = attn[..., :-1] @ v + attn[..., -1:] * xq_v
        else:
            out = attn @ v

        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class MultiHeadAttention(BaseMultiHeadAttention):
    @check_shapes(
        "xq: [m, nq, dqk]",
        "xk: [m, nkv, dqk]",
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
    ):
        return super().propagate(xq, xk, xv, mask)


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().propagate(x, x, x, mask)


class MultiHeadCrossAttention(BaseMultiHeadAttention):
    def __init__(
        self,
        *,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(qk_dim=embed_dim, v_dim=embed_dim, **kwargs)

    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dx]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        return super().propagate(xq, xkv, xkv, mask)
