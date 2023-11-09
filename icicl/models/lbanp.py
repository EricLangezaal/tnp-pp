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
from ..utils.helpers import preprocess_observations
from .base import NeuralProcess


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


class NestedLBANPDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("zt: [m, nt, dz]", "xt: [m, nt, dx]", "return: [m, nt, dy]")
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


class LBANP(NeuralProcess):
    def __init__(
        self,
        encoder: Union[LBANPEncoder, NestedLBANPEncoder],
        decoder: Union[LBANPDecoder, NestedLBANPDecoder],
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
