import string
import torch
from torch import nn
from abc import ABC
from typing import Union, Tuple, Optional
from check_shapes import check_shapes

from einops import rearrange
import warnings

from ..utils.conv import make_grid_from_range, unflatten_grid, flatten_grid, convNd
from ..models.ootg_tnp import OOTG_TNPDEncoder, OOTGSetConvTNPDEncoder, OOTG_MHCA_TNPDEncoder
from ..networks.attention_layers import MultiHeadSelfAttentionLayer
from ..networks.attention import MultiHeadSelfAttention


class WindowedMultiHeadSelfAttention(MultiHeadSelfAttentionLayer):

    def __init__(
            self, 
            *, 
            grid_range: Tuple[Tuple[float, float], ...],
            points_per_unit: int,
            window_sizes: Union[int, Tuple[int]],
            shift: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift = shift
        self.grid_shape = make_grid_from_range(grid_range, points_per_unit).shape[1:-1]
        self.window_sizes = torch.as_tensor((window_sizes,)).flatten()

        assert len(self.window_sizes) == len(self.grid_shape), "Please provide a window size for every grid dimension"

        self.num_windows = (torch.tensor(self.grid_shape) // self.window_sizes).to(int)
        assert torch.equal(self.num_windows, torch.tensor(self.grid_shape) / self.window_sizes), "Please make sure each grid dimension is divisible by its window size"

        n_strings, d_strings = [f'n{i}' for i in range(len(self.grid_shape))], [f'd{i}' for i in range(len(self.grid_shape))]
        paired = " ".join([f"({n} {d})" for n, d in zip(n_strings, d_strings)])
        self.reshape_pattern = f"b {paired} e -> (b {' '.join(n_strings)}) ({' '.join(d_strings)}) e"
        self.reshape_vars = dict(zip(d_strings, self.window_sizes))

    
    @check_shapes("x: [m, n, d]", "return: [m, n, d]")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        mask = None

        # from 'b num_ontg e -> b dim1 dim2 ... dim_n e'
        x = unflatten_grid(x, grid_shape=self.grid_shape)

        if self.shift:
            x = torch.roll(x, (-self.window_sizes // 2).tolist(), dims=torch.arange(1, x.dim() - 1).tolist())
            flat_size = self.window_sizes.prod().item()

            """
            My idea is that creating the mask is easier in 'unwrapped' space.
            In regular Swin, the mask for columns is quite difficult because it is constructed in flattened space,
            but before flattening it is the same as the simpler row_mask I believe.

            So maybe create everything unflattened and then flatten instead?
            """

            # TODO this is the final required shape, might be easier to start with different shape and reshape to this.
            mask = torch.zeros(batch_size * self.num_windows.prod().item(), flat_size, flat_size).to(x)
            
            # maybe this shape is easier?
            mask = torch.zeros(batch_size, *self.num_windows, flat_size, flat_size).to(x)

            # INCORRECT yet I think
            local_mask = torch.zeros(*self.window_sizes, *self.window_sizes).to(x)
            # do this for every dimension in window_size at the right location
            local_mask[-self.window_sizes[0]:, ..., self.window_sizes[0]:, ...] = -torch.inf
            local_mask[self.window_sizes[0]:, ..., -self.window_sizes[0]:, ...] = -torch.inf
            

             # for first dim you can get all halved patches this way now:
            mask[:, -1, ...] = local_mask.reshape(flat_size, -1)
            mask = mask.reshape(-1, flat_size, flat_size)



        # this reshapes: 'b (n0 d0) (n1 d1) (n2 d2) (..) e -> (b n0 n1 n2 .. ) (d0 d1 d2 ..) e'
        x = rearrange(x, self.reshape_pattern, **self.reshape_vars)

        x = super().forward(x, mask)
        x = rearrange(x, "(b ns) ds e -> b (ns ds) e", b=batch_size)
        return x


class SwinMultiHeadSelfAttentionLayer(MultiHeadSelfAttentionLayer):

    def __init__(self, **kwargs):
        nn.Module.__init__(self) # do not initialise parents as this layer should not have weights yet
        self.win_att_layer = WindowedMultiHeadSelfAttention(**kwargs)
        self.shifted_win_att_layer = WindowedMultiHeadSelfAttention(shift=True, **kwargs)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            warnings.warn("Swin Attention needs to construct its own mask, specified mask will not be used")
        
        x = self.win_att_layer(x)
        x = self.shifted_win_att_layer(x)
        return x




    
