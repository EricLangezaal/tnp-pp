import string
import torch
from torch import nn
from abc import ABC
from typing import Union, Tuple, Optional
from check_shapes import check_shapes

from einops import rearrange
import warnings

from ..utils.conv import make_grid, unflatten_grid, flatten_grid, convNd
from ..models.ootg_tnp import OOTG_TNPDEncoder, OOTGSetConvTNPDEncoder, OOTG_MHCA_TNPDEncoder
from ..networks.attention_layers import MultiHeadSelfAttentionLayer


class SwinMultiHeadSelfAttentionLayer(MultiHeadSelfAttentionLayer):

    def __init__(
            self, 
            *, 
            grid_range: Tuple[Tuple[float, float], ...],
            points_per_unit: int,
            window_sizes: Union[float, Tuple[float]],
            **kwargs,
    ):
        super().__init__(**kwargs)
        grid_shape = make_grid(grid_range[:, :1], grid_range[:, 1:2], points_per_unit, 0).shape[1:-1]
        self.window_sizes = window_sizes if isinstance(window_sizes, tuple) else (window_sizes, )
        num_windows = torch.tensor(grid_shape) / torch.tensor(self.window_sizes)

        # can only work if enough window sizes and if they properly divide the grid
        assert torch.equal(num_windows.to(int), num_windows)

        n_strings, d_strings = [f'n{i}' for i in range(len(grid_shape))], [f'd{i}' for i in range(len(grid_shape))]
        stacked = " ".join([f"{n} {d}" for n, d in zip(n_strings, d_strings)])

        self.pattern_unroll = f"b ({stacked}) e -> b {stacked} e"
        self.pattern_batch = f"b {stacked} e -> (b {' '.join(n_strings)}) ({' '.join(d_strings)}) e"
        self.unroll_vars = dict(zip(n_strings, num_windows.to(int))) | dict(zip(d_strings, self.window_sizes))
    
    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("Swin Attention needs to construct its own mask, specified mask will not be used")
        batch_size = x.shape[0]

        # idea: 'b (n0 d0 n1 d1 n2 d2 ..) e -> b n0 d0 n1 d1 n2 d2 .. e'
        x = rearrange(x, self.pattern_unroll, **self.unroll_vars)
        # TODO shift windows here with mask

        # idea 'b n0 d0 n1 d1 n2 d2 .. e -> (b n0 n1 n2 .. ) (d0 d1 d2 ..) e'
        x = rearrange(x, self.pattern_batch)
        super().forward(x, mask)
        
        x = rearrange(x, "(b ns) ds e -> b (ns ds) e", b=batch_size)
        return x





    
