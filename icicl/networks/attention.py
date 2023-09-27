"""https://github.com/rishikksh20/CrossViT-pytorch/blob/master/module.py"""

import einops
import torch
from check_shapes import check_shapes
from torch import nn


class MultiHeadSelfAttention(nn.Module):
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

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    @check_shapes("x: [m, n, d]", "return: [m, n, d]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Each of shape (m, num_heads, n, head_dim).
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = attn @ v
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out
