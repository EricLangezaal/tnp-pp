from typing import Tuple, Union
from abc import abstractmethod

import torch
from torch import nn

from .base import OOTGConditionalNeuralProcess
from .tnp import EfficientTNPDEncoder, TNPDDecoder
from ..utils.conv import compute_eq_weights, unflatten_grid, flatten_grid, convNd
from ..utils.helpers import preprocess_observations


class OOTGSetConvEncoder(nn.Module):
    def __init__(
        self,
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
    
    def forward(
            self, 
            xc_off_grid: torch.Tensor, 
            yc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            yc_on_grid: torch.Tensor
    ) -> Tuple[torch.Tensor]:
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
    

class OOTG_TNPDEncoder(EfficientTNPDEncoder):

    def __init__(
            self, 
            *,
            grid_encoder: Union[OOTGSetConvEncoder],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_encoder = grid_encoder

    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor
    ) -> torch.Tensor:
        xc, yc = self.grid_encoder(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)

        return super().forward(xc=xc, yc=yc, xt=xt)
    
    
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

    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor
    ) -> torch.Tensor:
        # this will make yc's last dimension 2.
        xc, yc = self.grid_encoder(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
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
