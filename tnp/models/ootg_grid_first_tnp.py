# This file is for reference only, to show that the OLD SetConvEncoder performs far worse
# This model joins the two grids first, before embedding/preprocessing which massively hurts performance.

from typing import Tuple, Union

import torch
from torch import nn

from .tnp import TNPDEncoder
from ..data.on_off_grid import DataModality
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..networks.attention import MultiHeadCrossAttention
from ..networks.grid_encoders import mhca_to_grid
from ..utils.conv import compute_eq_weights
from ..utils.grids import flatten_grid, make_grid_from_range


class OOTGSetConvTNPDEncoder(TNPDEncoder):
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

        xc_on_grid = flatten_grid(xc_on_grid)
        weights = compute_eq_weights(xc_on_grid, xc_off_grid, lengthscales=self.lengthscale)
        yc_off_grid_gridded = weights @ yc_off_grid
        
        # shape (batch_size, num_ontg, xdim)
        xc = xc_on_grid
        # shape (batch_size, num_ontg, 2)
        yc = torch.cat((flatten_grid(yc_on_grid), yc_off_grid_gridded), dim=-1)
        return xc, yc
    
    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        xc, yc = self.grid_encode(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
        if used_modality == DataModality.OFF_GRID:
            yc = yc[..., 1:]
        elif used_modality == DataModality.ON_GRID:
            yc = yc[..., :1]
        return TNPDEncoder.forward(self, xc=xc, yc=yc, xt=xt)
    

class OOTG_MHCA_TNPDEncoder(TNPDEncoder):

    def __init__(
            self,
            dim: int,
            grid_mhca_layer: Union[MultiHeadCrossAttentionLayer, MultiHeadCrossAttention],
            grid_range: Tuple[Tuple[float, float], ...] = None,
            points_per_unit: int = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.grid_mhca_layer = grid_mhca_layer
        self.latents = nn.Parameter(torch.randn(*make_grid_from_range(grid_range, points_per_unit).shape[:-1], dim))
        self.fake_embedding = nn.Parameter(torch.randn(dim))

    def grid_encode(
            self, 
            xc_off_grid: torch.Tensor, 
            yc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            yc_on_grid: torch.Tensor,
            used_modality: DataModality,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.latents.expand(xc_on_grid.shape[0], *self.latents.shape)

        yc = mhca_to_grid(xc_off_grid, yc_off_grid, 
                     xc_on_grid, yc_on_grid, 
                     latents, 
                     self.grid_mhca_layer, 
                     self.fake_embedding if used_modality == DataModality.OFF_GRID else None
        )
        return xc_on_grid, yc

    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        xc, yc = self.grid_encode(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid, used_modality=used_modality)
        return TNPDEncoder.forward(self, xc=xc, yc=yc, xt=xt)