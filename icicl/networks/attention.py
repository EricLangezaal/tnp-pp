"""https://github.com/rishikksh20/CrossViT-pytorch/blob/master/module.py"""
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, head_dim: int, p_dropout: float = 0.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == embed_dim)

        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dx]",
    )
    def forward(
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

        dots = (q @ k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            dots = torch.masked_fill(dots, mask, -float("Inf"))

        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class MultiHeadSelfAttention(MultiHeadAttention):
    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, x, x, mask)


class MultiHeadCrossAttention(MultiHeadAttention):
    @check_shapes(
        "xq: [m, nq, dx]",
        "xkv: [m, nkv, dx]",
        "mask: [m, nq, nkv]",
        "return: [m, n, d]",
    )
    def forward(
        self, xq: torch.Tensor, xkv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        return super().forward(xq, xkv, xkv, mask)
