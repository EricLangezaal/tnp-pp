from abc import ABC
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.group_actions import translation
from .kernels import Kernel


class MultiHeadTEAttention(nn.Module, ABC):
    def __init__(
        self,
        kernel: Kernel,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        p_dropout: float = 0.0,
        post_kernel: bool = False,
        group_action: Callable = translation,
        phi_t: Optional[nn.Module] = None,
        qk_dim: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = head_dim
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

        if qk_dim is not None and post_kernel:
            # Update inner dim to accommodate qk_dim.
            inner_dim = head_dim * qk_dim

        self.to_k = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)

        # Whether or not to pass through kernel after combination with inner products of tokens.
        self.post_kernel = post_kernel

        # Group action on inputs prior to kernel.
        self.group_action = group_action

        # Additional transformation on spatio-temporal locations.
        self.phi_t = phi_t

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nkv, dt]",
        "mask: [m, nq, nkv]",
        "return[0]: [m, nq, dx]",
        "return[1]: [m, nq, dt]",
    )
    def propagate(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Compute output of group action.
        # (m, nq, nkv, dx).
        diff = self.group_action(tq, tk)

        # Compute token attention.
        q = self.to_q(xq)
        k = self.to_k(xk)

        # Each of shape (m, {num_heads, qk_dim}, n, head_dim).
        q, k = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", d=self.head_dim),
            (q, k),
        )

        # (m, h, nq, nk).
        token_dots = (q @ k.transpose(-1, -2)) * self.scale

        if not self.post_kernel:
            # (m, {1, h}, nq, nkv).
            dots = self.kernel(diff, mask)
            dots = dots + token_dots
        else:
            token_dots = einops.rearrange(token_dots, "m h nq nk -> m nq nk h")
            kernel_input = torch.cat((diff, token_dots), dim=-1)
            dots = self.kernel(kernel_input, mask)

        attn = dots.softmax(dim=-1)

        # Multiply by values.
        v = self.to_v(xv)
        v = einops.rearrange(v, "b n (h d) -> b h n d", d=self.head_dim)
        out = attn @ v
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # Also update spatio-temporal locations if necessary.
        # Analgous to equation 4 in equivariant GNNs.
        if self.phi_t:
            attn = einops.rearrange(attn, "m h n p -> m n p h")
            t_dots = self.phi_t(attn)
            tq_new = tq + (1 / tk.shape[-2]) * (diff * t_dots).sum(-2)
        else:
            tq_new = tq

        return out, tq_new


class MultiHeadSelfTEAttention(MultiHeadTEAttention):
    @check_shapes(
        "x: [m, n, dx]",
        "t: [m, n, dt]",
        "mask: [m, n, n]",
        "return[0]: [m, n, dx]",
        "return[1]: [m, n, dt]",
    )
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(x, x, x, t, t, mask)


class MultiHeadCrossTEAttention(MultiHeadTEAttention):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().propagate(xq, xk, xk, tq, tk, mask)
