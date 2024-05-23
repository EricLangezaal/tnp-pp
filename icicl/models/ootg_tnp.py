from typing import Tuple, Union, Optional
from abc import abstractmethod

import torch
from torch import nn

from .base import OOTGConditionalNeuralProcess
from .tnp import EfficientTNPDEncoder, TNPDDecoder
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..utils.conv import compute_eq_weights, unflatten_grid, flatten_grid, convNd
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
        dim: int,
        init_lengthscale: float,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs) 
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

    """
    IDEA: zc_on_grid IS the start token
    Do one massive attention between zc_off_grid and zc_on_grid 
    with a mask that makes each on_grid point only look to closest off_grid points
    """
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
        num_latents = points_per_unit * (grid_range[:, 1] - grid_range[:, 0]).prod()
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.grid_mhca_layer = grid_mhca_layer

    def grid_encode(self, zc_off_grid: torch.Tensor, zc_on_grid: torch.Tensor) -> torch.Tensor:
        B, U, E = zc_off_grid.shape # 'U'nstructured
        S = zc_on_grid.shape[-2] # 'S'tructured

        u_expanded = zc_off_grid.repeat(S, 1, 1, 1).movedim(0, 1) # (B, S, U, E)
        s_expanded = zc_on_grid.repeat(U, 1, 1, 1).movedim(0, 2) # (B, S, U, E)
        
        u_idx = (u_expanded - s_expanded).abs().sum(dim=-1).argmin(dim=2).flatten()
        batch_idx = torch.arange(B).repeat_interleave(S)
        s_idx = torch.arange(S).repeat(B)

        mask = torch.zeros(B, S, U)
        mask[batch_idx, s_idx, u_idx] = 1

        


        mask = mask.to(zc_off_grid.device).to(torch.bool)

        zc = self.grid_mhca_layer(zc_off_grid, zc_on_grid, mask=mask)
        # shape as zc_on_grid so (B, S, E)
        return zc
       
    def forward( # picked apart to allow for embedding before gridding
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ):
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
    
    
class OOTG_ViTEncoder(OOTGSetConvTNPDEncoder):
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
        # this makes yc's last dimension 3.
        yc, yt = preprocess_observations(xt, yc) # 
        zc = torch.cat((xc, yc), dim=-1)
        # So zc is 3 + xdim
        zc = self.xy_encoder(zc)
        zc = self.coarsen_grid(zc)   

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        return self.transformer_encoder(zc, zt)



class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTG_TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
