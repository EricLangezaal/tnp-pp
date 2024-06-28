from typing import Tuple, Union, Optional

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import einops
from check_shapes import check_shapes

from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..networks.attention import MultiHeadCrossAttention
from ..utils.conv import compute_eq_weights
from ..utils.grids import unflatten_grid, flatten_grid, make_grid_from_range

class IdentityGridEncoder(nn.Module):

    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]"
    )
    def forward(
        self, 
        xc_off_grid: torch.Tensor, 
        xc_on_grid: torch.Tensor, 
        zc_off_grid: torch.Tensor, 
        zc_on_grid: torch.Tensor, 
        ignore_on_grid: bool,
    ) -> torch.Tensor:
        if ignore_on_grid:
            return zc_off_grid
        
        zc_on_grid = flatten_grid(zc_on_grid)
        zc = torch.cat((zc_off_grid, zc_on_grid), dim=-2)
        return zc
    

class SetConvGridEncoder(nn.Module):

    def __init__(
        self,
        *,
        dim: int,
        init_lengthscale: float,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__() 
        init_lengthscale = torch.as_tensor(dim * [init_lengthscale], dtype=dtype)
        self.lengthscale_param = nn.Parameter(
            (init_lengthscale.clone().detach().exp() - 1).log(),
            requires_grad=train_lengthscale,
        )

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )
    
    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]"
    )
    def forward(
        self, 
        xc_off_grid: torch.Tensor, 
        xc_on_grid: torch.Tensor, 
        zc_off_grid: torch.Tensor, 
        zc_on_grid: torch.Tensor, 
        ignore_on_grid: bool
    ) -> torch.Tensor:
        """
        Take a combination of the embedded on and the grid context data and merge those,
        by putting the off the grid data onto the same grid as the on-the grid,
        and then stacking the two.

        Returns:
           zc: both modalities embedded and on a grid OR off the grid only, depending on ignore_on_grid
        """
        grid_shape = xc_on_grid.shape[1:-1]
        xc_on_grid = flatten_grid(xc_on_grid)

        weights = compute_eq_weights(xc_on_grid, xc_off_grid, lengthscales=self.lengthscale)
        # shape (batch_size, num_ontg, embed_dim)
        zc = weights @ zc_off_grid
        
        if not ignore_on_grid:
            zc_on_grid = flatten_grid(zc_on_grid)
            zc += zc_on_grid

        return unflatten_grid(zc, grid_shape) 
    

class PseudoTokenGridEncoder(nn.Module):
    FAKE_TOKEN = -torch.inf

    def __init__(
            self,
            *,
            embed_dim: int,
            mhca_layer: Union[MultiHeadCrossAttentionLayer, MultiHeadCrossAttention],
            grid_range: Optional[Tuple[Tuple[float, float], ...]] = None,
            points_per_unit: Optional[int] = None,
            num_latents: Optional[int] = None,
    ):
        super().__init__()
        if grid_range is None or points_per_unit is None:
            assert isinstance(num_latents, int)
        else:
            num_latents = flatten_grid(make_grid_from_range(grid_range, points_per_unit)).size(-2)
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.fake_embedding = nn.Parameter(torch.randn(embed_dim))
        self.mhca_layer = mhca_layer

    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]"
    )
    def forward(
            self, 
            xc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            zc_off_grid: torch.Tensor, 
            zc_on_grid: torch.Tensor, 
            ignore_on_grid: bool
    ) -> torch.Tensor:
        grid_shape = torch.as_tensor(xc_on_grid.shape[1:-1], device=xc_on_grid.device)
        xc_on_grid = flatten_grid(xc_on_grid)
        zc_on_grid = flatten_grid(zc_on_grid)

        B, U, E = zc_off_grid.shape # 'U'nstructured
        S = zc_on_grid.shape[-2] # 'S'tructured

        # Quick calculation of nearest grid neighbour.
        x_grid_min = xc_on_grid.amin(dim=(0, 1))
        x_grid_max = xc_on_grid.amax(dim=(0, 1))
        x_grid_spacing = (x_grid_max - x_grid_min) / (grid_shape - 1)

        nearest_multi_idx = (xc_off_grid - x_grid_min + x_grid_spacing / 2) // x_grid_spacing
        nearest_multi_idx = torch.max(
            torch.min(nearest_multi_idx, grid_shape - 1), torch.zeros_like(grid_shape)
        )
        strides = torch.flip(
            torch.cumprod(
                torch.cat((torch.ones((1,), device=grid_shape.device), grid_shape), dim=0),
                dim=0,
            )[:-1],
            dims=(0,),
        )
        # (batch_size, U).
        nearest_idx = (nearest_multi_idx * strides).sum(dim=-1).type(torch.int)

        # shape (B, U)
        # _batch_ first repeats batch number then increments, range first increments then repeats
        u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
        u_range_idx = torch.arange(U).repeat(B, 1)

        # construct temporary mask that links each off-grid point to its closest on_grid point
        nearest_mask = torch.zeros(B, U, S, dtype=torch.int, device=zc_on_grid.device)
        nearest_mask[u_batch_idx, u_range_idx, nearest_idx] = 1

        # maximum nearest neigbours + add one for the on the grid point itself (or the placeholder)
        max_patch = nearest_mask.sum(dim=1).amax() + 1
        # batched cumulative count (i.e. add one if element has occured before)
        # So [0, 0, 2, 0, 1, 2] -> [0, 1, 0, 2, 0, 1] along each batch
        cumcount_idx = (nearest_mask.cumsum(dim=1) - 1)[u_batch_idx, u_range_idx, nearest_idx]

        # create tensor with for each grid-token all nearest off-grid + itself in third axis
        grid_stacked = torch.full((B * S, max_patch, E), self.FAKE_TOKEN, device=zc_on_grid.device)

        # add nearest off the grid points to each on_the_grid point
        idx_shifter = torch.arange(0, B * S, S).repeat_interleave(U)
        grid_stacked[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = (
             zc_off_grid[u_batch_idx.flatten(), u_range_idx.flatten()]
        )
        
        if ignore_on_grid:
            # add learned 'mask out' embedding at the end. Done instead of masking out in att_mask,
            # since that could potentially make it mask out whole input which crashes attention
            grid_stacked[torch.arange(B * S), -1] = self.fake_embedding
        else:
            # add the on_the_grid points themselves at the end
            s_batch_idx = torch.arange(B).repeat_interleave(S)
            s_range_idx = torch.arange(S).repeat(B)
            grid_stacked[torch.arange(B * S), -1] = zc_on_grid[s_batch_idx, s_range_idx]
    
        att_mask = torch.ones(B * S, 1, max_patch, dtype=torch.bool, device=zc_on_grid.device)
        # if fake token anywhere (sum with infinities stays infinity) in embedding, ignore it
        att_mask[(grid_stacked.sum(-1) == self.FAKE_TOKEN).unsqueeze(1)] = False

        # set fake value to something which won't overflow/NaN attention calculation
        grid_stacked[grid_stacked == self.FAKE_TOKEN] = 0

        # make latents of shape (?, E) to (B * S, 1, E)
        if self.latents.shape[0] == 1:
            latents = self.latents.expand((B * S, E)).unsqueeze(-2)
            #latents = einops.repeat(self.latents, "1 e -> (b s) 1 e", b=B, s=S)
        else:
            latents = einops.repeat(self.latents, "s e -> (b s) 1 e", b=B)
        
        # cannot use flash attention on our hardware, Mem efficient can't scale.
        # doesn't actually seem to hurt performance compared to mem efficient.
        with sdpa_kernel(SDPBackend.MATH):    
           zc = self.mhca_layer(latents, grid_stacked, mask=att_mask)
        # reshape output to match on_the_grid exactly again
        zc = einops.rearrange(zc, "(b s) 1 e -> b s e", b=B)

        # return zc in original shape (b, ..., e)
        return unflatten_grid(zc, tuple(grid_shape.tolist()))
