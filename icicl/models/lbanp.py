import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import NestedPerceiverEncoder
from .base import NeuralProcess


class LBANPEncoder(nn.Module):
    def __init__(
        self,
        nested_perceiver_encoder: NestedPerceiverEncoder,
        xy_encoder: nn.Module,
        x_encoder: nn.Module,
    ):
        super().__init__()

        self.nested_perceiver_encoder = nested_perceiver_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder

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
        zt = self.x_encoder(xt)

        zq = self.nested_perceiver_encoder(zc, zt)
        return zq


class LBANPDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("zq: [m, nt, dz]", "xt: [m, nt, dx]", "return: [m, nt, dy]")
    def forward(
        self,
        zt: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        _ = xt

        return self.z_decoder(zt)


class LBANP(NeuralProcess):
    def __init__(
        self,
        encoder: LBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
