from typing import Union

import torch
from torch import nn
from check_shapes import check_shapes

from .base import OOTGConditionalNeuralProcess
from .tnp import EfficientTNPDEncoder, TNPDDecoder

from ..networks.grid_encoders import IdentityGridEncoder, SetConvGridEncoder, PseudoTokenGridEncoder
from ..utils.grids import flatten_grid
from ..utils.helpers import preprocess_observations


class OOTG_TNPDEncoder(EfficientTNPDEncoder):

    def __init__(
            self,
            *,
            grid_encoder: Union[IdentityGridEncoder, SetConvGridEncoder, PseudoTokenGridEncoder],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.grid_encoder = grid_encoder

    @check_shapes("z: [b, ..., e]", "return: [b, n, e]")
    def prepare_context_tokens(
            self, 
            z: torch.Tensor
    ) -> torch.Tensor:
        return flatten_grid(z)
    
    @check_shapes(
        "xc_off_grid: [b, n, dx]", "yc_off_grid: [b, n, dy]", "xc_on_grid: [b, ..., dx]", "yc_on_grid: [b, ..., dy]", "xt: [b, nt, dx]"
    )
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

        zc = self.grid_encoder(
            xc_off_grid=xc_off_grid, xc_on_grid=xc_on_grid, 
            zc_off_grid=zc_off_grid, zc_on_grid=zc_on_grid,
            ignore_on_grid=ignore_on_grid
        )
        zc = self.prepare_context_tokens(zc)
        
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
