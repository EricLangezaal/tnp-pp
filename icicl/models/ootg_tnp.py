from typing import Tuple, Union, Optional
from abc import abstractmethod

import torch
from torch import nn
import einops
import warnings

from .base import OOTGConditionalNeuralProcess
from .tnp import EfficientTNPDEncoder, TNPDDecoder
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..utils.conv import compute_eq_weights, make_grid, unflatten_grid, flatten_grid, convNd
from ..utils.helpers import preprocess_observations

class OOTG_TNPDEncoder(EfficientTNPDEncoder):

    @abstractmethod
    def grid_encode(self, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError
    
    def forward(
            self, 
            xc_off_grid: torch.Tensor,
            yc_off_grid: torch.Tensor,
            xc_on_grid: torch.Tensor,
            yc_on_grid: torch.Tensor,
            xt: torch.Tensor,
            ignore_on_grid: bool = False,
    ): 
        raise NotImplementedError


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
            yc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            yc_on_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take a combination of on and the grid context data and merge those,
        by putting the off the grid data onto the same grid as the on-the grid,
        and then stacking the two.

        Returns:
           Tuple of merged xc and yc: both on a grid.
        """
        
        weights = compute_eq_weights(xc_on_grid, xc_off_grid, lengthscales=self.lengthscale)
        yc_off_grid_gridded = weights @ yc_off_grid
        
        # shape (batch_size, num_ontg, xdim)
        xc = xc_on_grid
        # shape (batch_size, num_ontg, 2)
        yc = torch.cat((yc_on_grid, yc_off_grid_gridded), dim=-1)
        return xc, yc
    
    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ):
        xc, yc = self.grid_encode(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
        if ignore_on_grid:
            yc = yc[..., 1:]
        return EfficientTNPDEncoder.forward(self, xc=xc, yc=yc, xt=xt)
    

class OOTG_MHCA_TNPDEncoder(OOTG_TNPDEncoder):
    # TODO: implement ignore_on_grid -> otherwise move to subclass only
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
        num_latents = make_grid(grid_range[:, :1], grid_range[:, 1:2], points_per_unit, 0).size(-2)
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.grid_mhca_layer = grid_mhca_layer

    def grid_encode(self, zc_off_grid: torch.Tensor, zc_on_grid: torch.Tensor) -> torch.Tensor:
        B, U, E = zc_off_grid.shape # 'U'nstructured
        S = zc_on_grid.shape[-2] # 'S'tructured

        s_expanded = zc_on_grid.repeat(U, 1, 1, 1).movedim(0, 1) # (B, U, S, E)
        u_expanded = zc_off_grid.repeat(S, 1, 1, 1).movedim(0, 2) # (B, U, S, E)
        
        nearest_idx = (u_expanded - s_expanded).abs().sum(dim=-1).argmin(dim=2)

        s_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, S)
        u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
        s_range_idx = torch.arange(S).repeat(B, 1)
        u_range_idx = torch.arange(U).repeat(B, 1)

        nearest_mask = torch.zeros(B, U, S, dtype=torch.int, device=zc_on_grid.device)
        nearest_mask[u_batch_idx, u_range_idx, nearest_idx] = 1

        # add one for the on the grid point
        max_patch = nearest_mask.sum(dim=1).amax() + 1
        # indexes show cumulative count (i.e. add one if element has occured before)
        cumcount_idx = (nearest_mask.cumsum(dim=1) - 1)[u_batch_idx, u_range_idx, nearest_idx]

        # create tensor with for each grid-token all nearest off-grid + itself in third dimension
        joint_grid = torch.full((B, S, max_patch, E), self.FAKE_TOKEN, device=zc_on_grid.device)
        joint_grid[u_batch_idx, nearest_idx, cumcount_idx] = zc_off_grid[u_batch_idx, torch.arange(U)]
        joint_grid[s_batch_idx, s_range_idx, -1] = zc_on_grid[s_batch_idx, torch.arange(S)]
        
        grid_stacked = einops.rearrange(joint_grid, "b s m e -> (b s) m e")

        att_mask = torch.ones(B * S, max_patch, S, dtype=torch.bool, device=zc_on_grid.device)
        # if fake token anywhere in embedding, ignore it
        att_mask[grid_stacked.sum(-1) == self.FAKE_TOKEN, :] = False

        # set fake value to something which won't overfloat attention calculation
        grid_stacked[grid_stacked == self.FAKE_TOKEN] = 0

        latents = self.latents.repeat(grid_stacked.size(0), 1, 1).to(zc_on_grid.device)
        zc = self.grid_mhca_layer(latents, grid_stacked, mask=att_mask.transpose(-1, -2))

        # TODO return shape is now (B, S, S, E) what do we do?
        zc = einops.rearrange(zc, "(b ignore) s e -> b ignore s e", b=B)
        # temporary
        return zc[:, 0]
       
    def forward( # picked apart to allow for embedding before gridding
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ):
        if ignore_on_grid:
            warnings.warn("Ignoring on the grid data not yet implemented for MHCA TNPD Encoder!")

        yc_off_grid, yt = preprocess_observations(xt, yc_off_grid)
        yc_on_grid, _ = preprocess_observations(xt, yc_on_grid)

        zc_off_grid = torch.cat((xc_off_grid, yc_off_grid), dim=-1)
        zc_off_grid = self.xy_encoder(zc_off_grid)
        zc_on_grid = torch.cat((xc_on_grid, yc_on_grid), dim=-1)
        zc_on_grid = self.xy_encoder(zc_on_grid)

        zc = self.grid_encode(zc_off_grid=zc_off_grid, zc_on_grid=zc_on_grid)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)
        return self.transformer_encoder(zc, zt)
    
    
class OOTG_ViTEncoder(OOTG_TNPDEncoder):
    """
    Implements a very basic ViT encoding without positional embeddings

    This relies on applying convolutions to coarsen the grid, which only works for grids that span up to 3 dimensions
    The dimensionality of the data is unrestricted.
    """

    def __init__(
            self,
            *,
            patch_size: int,
            dim: int,
            embed_dim: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.patcher = convNd(n=dim, in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)        

    def coarsen_grid(self, z: torch.Tensor) -> torch.Tensor:
        # z will be of shape (batch, num_on_grid, embed_dim)
        z = unflatten_grid(z, dim=self.dim)
        # move 'channels' (i.e embed_dim) right after batch
        z = z.movedim(-1, 1)
        z = self.patcher(z)
        # move 'channels' (i.e embed_dim) to end again
        z = z.movedim(1, -1)
        return flatten_grid(z)
    

class OOTGSetConvViTEncoder(OOTGSetConvTNPDEncoder, OOTG_ViTEncoder):
    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ) -> torch.Tensor:
        # this will make yc's last dimension 2.
        xc, yc = self.grid_encode(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
        if ignore_on_grid:
            yc = yc[..., 1:]
        # this makes yc's last dimension 3.
        yc, yt = preprocess_observations(xt, yc) # 
        zc = torch.cat((xc, yc), dim=-1)
        # So zc is 3 + xdim
        zc = self.xy_encoder(zc)
        zc = self.coarsen_grid(zc)   

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        return self.transformer_encoder(zc, zt)
    
    
class OOTG_MHCA_ViTEncoder(OOTG_MHCA_TNPDEncoder, OOTG_ViTEncoder):
    def grid_encode(self, zc_off_grid: torch.Tensor, zc_on_grid: torch.Tensor) -> torch.Tensor:
        zc = OOTG_MHCA_TNPDEncoder.grid_encode(zc_off_grid, zc_on_grid)
        zc = OOTG_ViTEncoder.coarsen_grid(zc)
        return zc


class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTG_TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
