import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import PerceiverDecoder, PerceiverEncoder
from .base import NeuralProcess


class PNPEncoder(nn.Module):
    def __init__(
        self,
        perceiver_encoder: PerceiverEncoder,
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
        _ = xt

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zq = self.perceiver_encoder(zc)
        return zq


class PNPDecoder(nn.Module):
    def __init__(
        self,
        perceiver_decoder: PerceiverDecoder,
        x_encoder: nn.Module,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.perceiver_decoder = perceiver_decoder
        self.x_encoder = x_encoder
        self.z_decoder = z_decoder

    @check_shapes("zq: [m, nq, dz]", "xt: [m, nt, dx]", "return: [m, nt, dy]")
    def forward(
        self,
        zq: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        zt = self.x_encoder(xt)
        zt = self.perceiver_decoder(zt, zq)

        return self.z_decoder(zt)


class PNP(NeuralProcess):
    def __init__(
        self,
        encoder: PNPEncoder,
        decoder: PNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
