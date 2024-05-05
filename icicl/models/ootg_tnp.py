from typing import Tuple, Union
from abc import abstractmethod

import torch
from torch import nn
import einops

from .base import OOTGConditionalNeuralProcess
from .tnp import TNPDEncoder, TNPDDecoder, gen_tnpd_mask
from ..utils.conv import compute_eq_weights
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
            (torch.tensor(init_lengthscale).exp() - 1).log(),
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
    

class OOTG_TNPDEncoder(TNPDEncoder):

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

    def __init__(
            self,
            *,
            patch_size: int,
            in_dim_x: int,
            out_dim_x: int,
            out_dim_y: int,
            in_dim_y: int = 2,
            **kwargs
    ):
         super().__init__(**kwargs)
         # TODO this assumes xdim is one, since my grid is just flattened...
         # Would have to use convNd otherwise?
         self.xpatcher = nn.Conv1d(in_channels=in_dim_x, out_channels=out_dim_x, kernel_size=patch_size, stride=patch_size)
         self.ypatcher = nn.Conv1d(in_channels=in_dim_y, out_channels=out_dim_y, kernel_size=patch_size, stride=patch_size)

    def coarsen_grid(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = einops.rearrange(x, "b n d -> b d n")
        x = self.xpatcher(x)
        x = einops.rearrange(x, "b e n -> b n e")

        y = einops.rearrange(y, "b n d -> b d n")
        y = self.ypatcher(y)
        y = einops.rearrange(y, "b e n -> b n e")
        return x, y

    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor
    ) -> torch.Tensor:
        xc, yc = self.grid_encoder(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
        xc, yc = self.coarsen_grid(x=xc, y=yc)

        return TNPDEncoder.forward(self, xc=xc, yc=yc, xt=xt)



class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTG_TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
