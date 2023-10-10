from abc import ABC
from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn

from .kernels import Kernel


class MultiHeadTEAttention(nn.Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        embed_dim: int,
        head_dim: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        project_out = not head_dim == embed_dim

        self.kernel = kernel
        self.to_v = nn.Linear(embed_dim, head_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(head_dim, embed_dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "yv: [m, nkv, dy]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dy]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        yv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xq_ = xq[:, :, None, :]
        xk_ = xk[:, None, :, :]

        # Compute pairwise differences.
        # (m, nq, nkv, dx).
        diff = xq_ - xk_

        # (m, nq, nkv).
        attn = self.kernel(diff, mask)

        # (m, nkv, head_dim).
        v = self.to_v(yv)

        out = attn @ v
        out = self.to_out(out)
        return out


class MultiHeadSelfTEAttention(MultiHeadTEAttention):
    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]", "mask: [m, n, n]", "return: [m, n, dy]"
    )
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().propagate(x, x, y, mask)


class MultiHeadCrossTEAttention(MultiHeadTEAttention):
    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "yv: [m, nkv, dy]",
        "mask: [m, nq, nkv]",
        "return: [m, nq, dy]",
    )
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        yv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        return super().propagate(xq, xk, yv, mask)
