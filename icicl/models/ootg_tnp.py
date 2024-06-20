from typing import Tuple

import torch
from torch import nn
import einops
from check_shapes import check_shapes

from .base import OOTGConditionalNeuralProcess
from .tnp import EfficientTNPDEncoder, TNPDDecoder
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..utils.conv import compute_eq_weights, make_grid, flatten_grid
from ..utils.helpers import preprocess_observations


class OOTG_TNPDEncoder(EfficientTNPDEncoder):

    def grid_encode(
        self, 
        xc_off_grid: torch.Tensor, 
        xc_on_grid: torch.Tensor, 
        zc_off_grid: torch.Tensor, 
        zc_on_grid: torch.Tensor, 
        ignore_on_grid: bool,
    ) -> torch.Tensor:
        if ignore_on_grid:
            return zc_off_grid
        
        zc = torch.cat((zc_off_grid, zc_on_grid), dim=-2)
        return zc
    
    def forward(
        self, 
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ) -> torch.Tensor:
        yc_off_grid, yt = preprocess_observations(xt, yc_off_grid, on_grid=False)
        yc_on_grid, _ = preprocess_observations(xt, yc_on_grid, on_grid=True)

        zc_off_grid = torch.cat((xc_off_grid, yc_off_grid), dim=-1)
        zc_off_grid = self.xy_encoder(zc_off_grid)
        zc_on_grid = torch.cat((xc_on_grid, yc_on_grid), dim=-1)
        zc_on_grid = self.xy_encoder(zc_on_grid)

        zc = self.grid_encode(
            xc_off_grid=xc_off_grid, xc_on_grid=xc_on_grid, 
            zc_off_grid=zc_off_grid, zc_on_grid=zc_on_grid,
            ignore_on_grid=ignore_on_grid
        )
        
        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)
        return self.transformer_encoder(zc, zt)


class OOTGSetConvTNPDEncoder(OOTG_TNPDEncoder):
    def __init__(
        self,
        *,
        dim_lengthscale: int,
        init_lengthscale: float,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs) 
        init_lengthscale = torch.as_tensor(dim_lengthscale * [init_lengthscale], dtype=dtype)
        self.lengthscale_param = nn.Parameter(
            (init_lengthscale.clone().detach().exp() - 1).log(),
            requires_grad=train_lengthscale,
        )

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )
    
    def grid_encode(
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
        
        weights = compute_eq_weights(xc_on_grid, xc_off_grid, lengthscales=self.lengthscale)
        # shape (batch_size, num_ontg, embed_dim)
        zc = weights @ zc_off_grid
        
        if not ignore_on_grid:
            zc += zc_on_grid
        return zc
    

class OOTG_MHCA_TNPDEncoder(OOTG_TNPDEncoder):
    FAKE_TOKEN = -torch.inf

    def __init__(
            self,
            *,
            embed_dim: int,
            grid_range: Tuple[Tuple[float, float], ...],
            points_per_unit: int,
            grid_mhca_layer: MultiHeadCrossAttentionLayer,
            **kwargs,
    ):
        super().__init__(**kwargs)
        grid_range = torch.as_tensor(grid_range)
        num_latents = flatten_grid(make_grid(grid_range[:, :1], grid_range[:, 1:2], points_per_unit, 0)).size(-2)
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.fake_embedding = nn.Parameter(torch.randn(embed_dim))

        self.grid_mhca_layer = grid_mhca_layer
        assert grid_mhca_layer.embed_dim == embed_dim, "embed_dim must match."

    @check_shapes(
        "xc_off_grid: [b, u, dx]", "xc_on_grid: [b, s, dx]", 
        "zc_off_grid: [b, u, e]", "zc_on_grid: [b, s, e]", 
        "return: [b, s, e]"
    )
    def grid_encode(
            self, 
            xc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            zc_off_grid: torch.Tensor, 
            zc_on_grid: torch.Tensor, 
            ignore_on_grid: bool
    ) -> torch.Tensor:
        
        B, U, E = zc_off_grid.shape # 'U'nstructured
        S = zc_on_grid.shape[-2] # 'S'tructured

        xc_on_grid = xc_on_grid.repeat(U, 1, 1, 1).movedim(0, 1) # (B, U, S, Xdim)
        xc_off_grid = xc_off_grid.repeat(S, 1, 1, 1).movedim(0, 2) # (B, U, S, Xdim)
        
        # shape (B, U): for each off_grid, tell me index of closest on_grid in same batch
        nearest_idx = (xc_off_grid - xc_on_grid).abs().sum(dim=-1).argmin(dim=2)

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

        # repeat latents of shape (S, E) to (B * S, 1, E)
        latents = self.latents.repeat(B, 1).unsqueeze(1).to(zc_on_grid.device)
        zc = self.grid_mhca_layer(latents, grid_stacked, mask=att_mask)
        # reshape output to match on_the_grid exactly again
        zc = einops.rearrange(zc, "(b s) 1 e -> b s e", b=B)
        return zc
    

class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTG_TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
