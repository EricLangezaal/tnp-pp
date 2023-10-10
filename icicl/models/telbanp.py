import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.tetransformer import NestedTEPerceiverEncoder
from .base import NeuralProcess
from .lbanp import LBANPDecoder


class TELBANPEncoder(nn.Module):
    def __init__(
        self,
        nested_perceiver_encoder: NestedTEPerceiverEncoder,
        y_encoder: nn.Module,
    ):
        super().__init__()

        self.nested_perceiver_encoder = nested_perceiver_encoder
        self.y_encoder = y_encoder

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

        zc = self.y_encoder(yc)
        zt = self.y_encoder(yt)

        zt = self.nested_perceiver_encoder(xc, xt, zc, zt)
        return zt


class TELBANP(NeuralProcess):
    def __init__(
        self,
        encoder: TELBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
