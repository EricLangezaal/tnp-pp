import itertools
import warnings
from typing import Optional, Tuple

import einops
import numpy as np
import torch
from check_shapes import check_shapes

from .attention_layers import MultiHeadSelfAttentionLayer


class SWINAttentionLayer(MultiHeadSelfAttentionLayer):
    def __init__(
        self,
        *,
        window_sizes: Tuple[int, ...],
        shift_sizes: Optional[Tuple[int, ...]] = None,
        roll_dims: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask = torch.empty(0) # dummy will be overwritten
        self.window_sizes = torch.as_tensor(window_sizes)

        if shift_sizes is not None:
            self.shift_sizes = torch.as_tensor(shift_sizes)
        else:
            self.shift_sizes = self.window_sizes // 2
        
        self.roll_dims = roll_dims

    @check_shapes("x: [m, ..., d]", "mask: [m, ...]", "return: [m, ..., d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn(
                "Swin Attention needs to construct its own mask, specified mask will not be used."
            )

        num_batches = x.shape[0]
        grid_shape = torch.as_tensor(x.shape[1:-1], dtype=int)

        assert torch.all(grid_shape % self.window_sizes == 0), "grid must be divisible by window size"

        # First no shift.
        x = window_partition(x, self.window_sizes)
        # Combine batch dimensions for efficient computation.
        x = einops.rearrange(x, "m nw ws d -> (m nw) ws d")
        x = super().forward(x)
        x = einops.rearrange(x, "(m nw) ws d -> m nw ws d", m=num_batches)
        x = window_reverse(x, self.window_sizes, grid_shape)

        # Now shift.
        shifted_x = torch.roll(
            x,
            shifts=(-self.shift_sizes).tolist(),
            dims=list(range(1, len(self.shift_sizes) + 1)),
        )
        shifted_x = window_partition(shifted_x, self.window_sizes)

        if self.mask.shape[0] != (grid_shape // self.window_sizes).prod():
            # Compute attention mask for shifted windows.
            self.mask = swin_attention_mask(
                self.window_sizes,
                self.shift_sizes,
                grid_shape,
                roll_dims=self.roll_dims,
                device=x.device,
            )
        # Combine batch dimensions for efficient computation.
        mask = einops.repeat(self.mask, "nw ws1 ws2 -> (m nw) ws1 ws2", m=num_batches)

        shifted_x = einops.rearrange(shifted_x, "m nw ws d -> (m nw) ws d")
        shifted_x = super().forward(shifted_x, mask=mask)
        shifted_x = einops.rearrange(shifted_x, "(m nw) ws d -> m nw ws d", m=num_batches)
        shifted_x = window_reverse(shifted_x, self.window_sizes, grid_shape)

        # Unshift.
        x = torch.roll(
            shifted_x,
            shifts=(self.shift_sizes).tolist(),
            dims=list(range(1, len(self.shift_sizes) + 1)),
        )
        return x


@check_shapes(
    "x: [m, ..., d]",
    "return: [m, nw, ws, d]",
)
def window_partition(x: torch.Tensor, window_sizes: torch.Tensor):
    grid_shape = x.shape[1:-1]

    n_strings, d_strings = [f"n{i}" for i in range(len(grid_shape))], [
        f"d{i}" for i in range(len(grid_shape))
    ]
    paired = " ".join([f"({n} {d})" for n, d in zip(n_strings, d_strings)])
    reshape_pattern = (
        f"b {paired} e -> b ({' '.join(n_strings)}) ({' '.join(d_strings)}) e"
    )
    reshape_vars = dict(zip(d_strings, window_sizes))

    return einops.rearrange(x, reshape_pattern, **reshape_vars)


@check_shapes(
    "x: [m, nw, ws, d]",
    "return: [m, ..., d]",
)
def window_reverse(
    x: torch.Tensor, window_sizes: torch.Tensor, grid_shape: torch.Tensor
):
    num_windows = grid_shape // window_sizes
    n_strings, d_strings = [f"n{i}" for i in range(len(grid_shape))], [
        f"d{i}" for i in range(len(grid_shape))
    ]
    paired = " ".join([f"({n} {d})" for n, d in zip(n_strings, d_strings)])
    unreshape_pattern = (
        f"b ({' '.join(n_strings)}) ({' '.join(d_strings)}) e -> b {paired} e"
    )
    window_size_vars = dict(zip(d_strings, window_sizes))
    num_windows_vars = dict(zip(n_strings, num_windows))

    return einops.rearrange(x, unreshape_pattern, **window_size_vars | num_windows_vars)


def swin_attention_mask(
    window_sizes: torch.Tensor,
    shift_sizes: torch.Tensor,
    grid_shape: torch.Tensor,
    roll_dims: Optional[Tuple[int, ...]] = None,
    device: str = "cpu",
):
    img_mask = torch.ones((1, *grid_shape, 1), device=device)

    slices = [
        (
            slice(0, -window_sizes[i]),
            slice(-window_sizes[i], -shift_sizes[i]),
            slice(-shift_sizes[i], None),
        )
        for i in range(len(grid_shape))
    ]

    if roll_dims is not None:
        for dim in roll_dims:
            slices[dim] = (
                slice(0, -window_sizes[dim]),
                slice(-window_sizes[dim], None),
            )

    cnt = 0
    for slices_ in itertools.product(*slices):
        slices_ = (slice(None), *slices_, slice(None))
        img_mask[slices_] = cnt
        cnt += 1

    # (1, num_windows, tokens_per_window).
    mask_windows = window_partition(img_mask, window_sizes).squeeze(-1)

    # (num_windows, tokens_per_window, tokens_per_window).
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -np.inf).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask[0]