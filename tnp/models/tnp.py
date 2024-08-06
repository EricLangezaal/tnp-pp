from typing import Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TNPDTransformerEncoder, TransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess, NeuralProcess


class TNPDEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TNPDTransformerEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = torch.cat((xc, yc), dim=-1)
        zt = torch.cat((xt, yt), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        zt = self.transformer_encoder(zc, zt)
        return zt


class TNPDDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("z: [m, ..., n, dz]", "xt: [m, nt, dx]", "return: [m, ..., nt, dy]")
    def forward(self, z: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        zt = z[..., -xt.shape[-2] :, :]
        return self.z_decoder(zt)


class TNPD(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)