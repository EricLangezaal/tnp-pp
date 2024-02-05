import copy
from abc import ABC
from typing import Callable, Optional, Tuple, Union

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
        add_diagonal_attention: bool = False,
        phi_t_reuse_attn: bool = True,
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

        # Whether to do diagonals in mhca attention.
        self.add_diagonal_attention = add_diagonal_attention

        # Whether to use a separate attention mechanism for phi_t inputs.
        if not phi_t_reuse_attn:
            self.phi_t_to_q = nn.Linear(embed_dim, inner_dim, bias=False)
            self.phi_t_to_k = nn.Linear(embed_dim, inner_dim, bias=False)
            self.phi_t_kernel = copy.deepcopy(self.kernel)
        else:
            self.phi_t_to_q = self.phi_t_to_k = self.phi_t_kernel = None

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

        # Compute attention weights used for token update.
        attn, out = self._get_attn_weights(
            xq=xq,
            xk=xk,
            xv=xv,
            tq=tq,
            tk=tk,
            mask=mask,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            to_out=self.to_out,
            kernel=self.kernel,
            add_diagonal_attention=self.add_diagonal_attention,
            return_out=True,
        )

        # Also update spatio-temporal locations if necessary.
        if self.phi_t:
            if self.phi_t_to_q is not None:
                phi_t_input = self._get_attn_weights(
                    xq=xq,
                    xk=xk,
                    xv=xv,
                    tq=tq,
                    tk=tk,
                    mask=mask,
                    to_q=self.phi_t_to_q,
                    to_k=self.phi_t_to_k,
                    kernel=self.phi_t_kernel,
                    return_out=False,
                )
            else:
                phi_t_input = attn
                if self.add_diagonal_attention:
                    # Remove diagonal attention bit.
                    phi_t_input = phi_t_input[..., :-1]

            phi_t_input = einops.rearrange(phi_t_input, "m h n p -> m n p h")
            t_dots = self.phi_t(phi_t_input)
            tq_new = tq + (diff * t_dots).sum(-2)
        else:
            tq_new = tq

        return out, tq_new

    @check_shapes(
        "xq: [m, nq, dx]",
        "xk: [m, nkv, dx]",
        "xv: [m, nkv, dx]",
        "tq: [m, nq, dt]",
        "tk: [m, nkv, dt]",
        "mask: [m, nq, nkv]",
        # "return: [m, nq, ...]",
    )
    def _get_attn_weights(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        tq: torch.Tensor,
        tk: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        to_q: Optional[nn.Module] = None,
        to_k: Optional[nn.Module] = None,
        to_v: Optional[nn.Module] = None,
        to_out: Optional[nn.Module] = None,
        kernel: Optional[nn.Module] = None,
        add_diagonal_attention: bool = False,
        return_out: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Computes multi-head translation equivariant attention weights.

        Args:
            xq (torch.Tensor): Query token.
            xk (torch.Tensor): Key token.
            xv (torch.Tensor): Value token.
            tq (torch.Tensor): Query inputs.
            tk (torch.Tensor): Key inputs.
            mask (Optional[torch.Tensor], optional): Query-key mask. Defaults to None.

        Returns:
            torch.Tensor: attention weights..
        """
        to_q = self.to_q if to_q is None else to_q
        to_k = self.to_k if to_k is None else to_k
        kernel = self.kernel if kernel is None else kernel

        # Compute output of group action.
        # (m, nq, nkv, dx).
        diff = self.group_action(tq, tk)

        # Compute token attention.
        q = to_q(xq)
        k = to_k(xk)

        # Each of shape (m, {num_heads, qk_dim}, n, head_dim).
        q, k = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", d=self.head_dim),
            (q, k),
        )

        # (m, h, nq, nk).
        token_dots = (q @ k.transpose(-1, -2)) * self.scale

        if add_diagonal_attention:
            assert (
                xq.shape[-1] == xv.shape[-1]
            ), "xq and xv must have same embedding dimension."
            xq_k = to_k(xq)
            xq_k = einops.rearrange(xq_k, "b n (h d) -> b h n d", d=self.head_dim)
            # Add diagonal self attention amongst xq.
            diag_token_dots = (q * xq_k).sum(-1, keepdim=True) * self.scale

            # (m, h, nq, nk + 1).
            token_dots = torch.cat((token_dots, diag_token_dots), dim=-1)

            # Add diagonal diffs to diff.
            diag_diff = self.group_action(tq, tq, diagonal=True).unsqueeze(-2)
            diff = torch.cat((diff, diag_diff), dim=-2)

        if not self.post_kernel:
            # (m, {1, h}, nq, nkv).
            dots = kernel(diff)
            dots = einops.rearrange(dots, "m nq nk h -> m h nq nk")
            dots = dots + token_dots
        else:
            token_dots = einops.rearrange(token_dots, "m h nq nk -> m nq nk h")
            kernel_input = torch.cat((diff, token_dots), dim=-1)
            dots = kernel(kernel_input)
            dots = einops.rearrange(dots, "m nq nk h -> m h nq nk")

        if mask is not None:
            mask = einops.repeat(mask, "m n p -> m h n p", h=self.num_heads)
            dots = torch.masked_fill(dots, mask, -float("inf"))

        # (m, num_heads, nq, nk).
        attn = dots.softmax(dim=-1)

        if return_out:
            to_v = self.to_v if to_v is None else to_v
            to_out = self.to_out if to_out is None else to_out
            v = to_v(xv)
            v = einops.rearrange(v, "b n (h d) -> b h n d", d=self.head_dim)

            if add_diagonal_attention:
                xq_v = to_v(xq)
                xq_v = einops.rearrange(xq_v, "b n (h d) -> b h n d", d=self.head_dim)
                out = attn[..., :-1] @ v + attn[..., -1:] * xq_v
            else:
                out = attn @ v

            out = einops.rearrange(out, "b h n d -> b n (h d)")
            out = self.to_out(out)
            return attn, out

        return attn


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
