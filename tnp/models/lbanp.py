from typing import Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import (
    ISetTransformerEncoder,
    NestedISetTransformerEncoder,
    NestedPerceiverEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
)

from ..data.on_off_grid import DataModality
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess


class NestedLBANPEncoder(nn.Module):
    def __init__(
        self,
        perceiver_encoder: Union[NestedPerceiverEncoder, NestedISetTransformerEncoder],
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.perceiver_encoder = perceiver_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, nq, dz]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.perceiver_encoder(zc, zt)
        return zt


class OOTGNestedLBANPEncoder(NestedLBANPEncoder):
    def __init__(
            self,
            x_encoder: nn.Module = nn.Identity(),
            y_encoder: nn.Module = nn.Identity(),
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

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

        # join modalities and encode (i.e. Sirennet etc)
        xc = used_modality.get(xc_on_grid, xc_off_grid)
        xc = self.x_encoder(xc)

        yc = used_modality.get(yc_on_grid, yc_off_grid)
        yc = self.y_encoder(yc)

        xt = self.x_encoder(xt)
        yt = self.y_encoder(yt)

        # --- copied from NestedLBANPEncoder ---
        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.perceiver_encoder(zc, zt)
        return zt


class NestedLBANPDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("zt: [m, ..., nt, dz]", "xt: [m, nt, dx]", "return: [m, ..., nt, dy]")
    def forward(
        self,
        zt: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        _ = xt

        return self.z_decoder(zt)


class LBANPEncoder(nn.Module):
    def __init__(
        self,
        perceiver_encoder: Union[PerceiverEncoder, ISetTransformerEncoder],
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.perceiver_encoder = perceiver_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return[0]: [m, nt, dz]",
        "return[1]: [m, nq, dz]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        zq = self.perceiver_encoder(zc)
        return zt, zq


class LBANPDecoder(nn.Module):
    def __init__(
        self,
        perceiver_decoder: PerceiverDecoder,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.perceiver_decoder = perceiver_decoder
        self.z_decoder = z_decoder

    @check_shapes(
        "tokens[0]: [m, nt, dz]",
        "tokens[1]: [m, nq, dz]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dy]",
    )
    def forward(
        self,
        tokens: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        _ = xt

        zt, zq = tokens
        zt = self.perceiver_decoder(zt, zq)

        return self.z_decoder(zt)


class LBANP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: Union[LBANPEncoder, NestedLBANPEncoder],
        decoder: Union[LBANPDecoder, NestedLBANPDecoder],
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
