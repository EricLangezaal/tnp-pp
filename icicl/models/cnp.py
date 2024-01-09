import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.deepset import DeepSet
from .base import NeuralProcess


class CNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
    ):
        self.deepset = deepset

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        zc = self.deepset(xc, yc)
        # Use same context representation for every target point.
        zc = zc.unsqueeze(-2).repeat(1, xt.shape[-2], 1)
        return zc


class CNPDecoder(nn.Module):
    """Represents the decoder for a CNP."""

    def __init__(self, z_decoder: nn.Module):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes(
        "zc: [m, nt, dz]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(self, zc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        return self.z_decoder(torch.cat((zc, xt), dim=-1))


class CNP(NeuralProcess):
    def __init__(
        self,
        encoder: CNPEncoder,
        decoder: CNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
