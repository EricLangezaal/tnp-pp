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
    ):
        super().__init__()

        self.nested_perceiver_encoder = nested_perceiver_encoder
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
        yt = torch.zeros(xt.shape[:-1] + yc.shape[-1:])
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,))), dim=-1)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,))), dim=-1)

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.nested_perceiver_encoder(zc, zt)
        return zt


class LBANPDecoder(nn.Module):
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


class LBANP(NeuralProcess):
    def __init__(
        self,
        encoder: LBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
