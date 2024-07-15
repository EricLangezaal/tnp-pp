from typing import Union, Callable

import torch
from torch import nn
from check_shapes import check_shapes

from .base import OOTGConditionalNeuralProcess
from .tnp import TNPDDecoder

from ..data.on_off_grid import DataModality
from ..networks.grid_encoders import IdentityGridEncoder, SetConvGridEncoder, PseudoTokenGridEncoder
from ..networks.grid_transformer import GridTransformerEncoder
from ..utils.grids import coarsen_grid
from ..utils.helpers import preprocess_observations


class OOTG_TNPDEncoder(nn.Module):

    def __init__(
            self,
            transformer_encoder: Union[GridTransformerEncoder],
            grid_encoder: Union[IdentityGridEncoder, SetConvGridEncoder, PseudoTokenGridEncoder],
            xy_encoder: nn.Module,
            x_encoder: nn.Module = nn.Identity(),
            y_encoder: nn.Module = nn.Identity(),
            patch_encoder: nn.Module = None,
            coarsen_fn: Callable = coarsen_grid,
    ):
        super().__init__()

        if type(grid_encoder) == IdentityGridEncoder:
            assert type(transformer_encoder) == GridTransformerEncoder and transformer_encoder.patch_encoder is None

        self.transformer_encoder = transformer_encoder
        self.grid_encoder = grid_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        self.patch_encoder = patch_encoder
        self.coarsen_fn = coarsen_fn

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
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        # add flag dimensions to all y values
        yc_off_grid, yt = preprocess_observations(xt, yc_off_grid, on_grid=False)
        yc_on_grid, _ = preprocess_observations(xt, yc_on_grid, on_grid=True)

        # encode OFF grid X data and separate it again
        x_off_grid = torch.cat((xc_off_grid, xt), dim=1)
        x_encoded = self.x_encoder(x_off_grid)
        xc_off_grid_encoded, xt_encoded = x_encoded.split((xc_off_grid.shape[1], xt.shape[1]), dim=1)

        # encode OFF grid Y data and separate it again
        y_off_grid = torch.cat((yc_off_grid, yt), dim=1)
        y_encoded = self.y_encoder(y_off_grid)
        yc_off_grid_encoded, yt_encoded = y_encoded.split((yc_off_grid.shape[1], yt.shape[1]), dim=1)

        # encode GRIDDED data
        xc_grid_encoded = self.x_encoder(xc_on_grid)
        yc_grid_encoded = self.y_encoder(yc_on_grid)

        # merge OFF grid context x and y and encode it
        zc_off_grid = torch.cat((xc_off_grid_encoded, yc_off_grid_encoded), dim=-1)
        zc_off_grid = self.xy_encoder(zc_off_grid)
        # merge ON grid context x and y and encode it
        zc_on_grid = torch.cat((xc_grid_encoded, yc_grid_encoded), dim=-1)
        zc_on_grid = self.xy_encoder(zc_on_grid)
        
        # PATCH Embed first if configured
        if self.patch_encoder is not None:
           zc_on_grid = self.patch_encoder(zc_on_grid)
           xc_on_grid = self.coarsen_fn(xc_on_grid, self.patch_encoder.conv.kernel_size)

        # merge OFF grid TARGET x and y and encode it
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zt = self.xy_encoder(zt)

        # encode context set!
        xc, zc = self.grid_encoder(
            xc_off_grid=xc_off_grid, xc_on_grid=xc_on_grid, 
            zc_off_grid=zc_off_grid, zc_on_grid=zc_on_grid,
            used_modality=used_modality
        )

        zt = self.transformer_encoder(xc, zc, xt, zt)
        return zt
    

class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTG_TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
