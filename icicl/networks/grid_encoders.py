from typing import Tuple, Union, Optional

import torch
from torch import nn
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
    
    # TODO can it work without flattening?
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
            **kwargs,
    ):
        super().__init__(**kwargs)
        if grid_range is None or points_per_unit is None:
            assert isinstance(num_latents, int)
            num_latents = num_latents
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
        grid_shape = xc_on_grid.shape[1:-1]
        xc_on_grid = flatten_grid(xc_on_grid)
        zc_on_grid = flatten_grid(zc_on_grid)

        B, U, E = zc_off_grid.shape # 'U'nstructured
        S = zc_on_grid.shape[-2] # 'S'tructured

        # TODO: there's a way of doing this which is linear, not quadratic.
        # Index of closest grid point to each point.
        # shape (B, U): for each off_grid, tell me index of closest on_grid in same batch
        nearest_idx = (
            (xc_off_grid[..., None, :] - xc_on_grid[:, None, ...]).abs().sum(dim=-1).argmin(dim=2)
        )

        # shape (B, U) or (B, S) respectively
        # _batch_ first repeats batch number then increments, range first increments then repeats
        s_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, S)
        u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
        s_range_idx = torch.arange(S).repeat(B, 1)
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
        joint_grid = torch.full((B, S, max_patch, E), self.FAKE_TOKEN, device=zc_on_grid.device)
        # add nearest off the grid points to each on_the_grid point
        joint_grid[u_batch_idx, nearest_idx, cumcount_idx] = zc_off_grid[u_batch_idx, torch.arange(U)]
        if ignore_on_grid:
            # add learned 'mask out' embedding at the end. Done instead of masking out in att_mask,
            # since that could potentially make it mask out whole input which crashes attention
            joint_grid[s_batch_idx, s_range_idx, -1] = self.fake_embedding
        else:
            # add the on_the_grid points themselves at the end
            joint_grid[s_batch_idx, s_range_idx, -1] = zc_on_grid[s_batch_idx, torch.arange(S)]
        
        grid_stacked = einops.rearrange(joint_grid, "b s m e -> (b s) m e")

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

        zc = self.mhca_layer(latents, grid_stacked, mask=att_mask)
        # reshape output to match on_the_grid exactly again
        zc = einops.rearrange(zc, "(b s) 1 e -> b s e", b=B)

        # return zc in original shape (b, ..., e)
        return unflatten_grid(zc, grid_shape)
    

