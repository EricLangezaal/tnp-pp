import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.tetransformer import NestedTEPerceiverEncoder
from ..utils.helpers import preprocess_observations
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
        yc, yt = preprocess_observations(xt, yc)

        zc = self.y_encoder(yc)
        zt = self.y_encoder(yt)

        zt = self.nested_perceiver_encoder(zc, zt, xc, xt)
        return zt


class TELBANP(NeuralProcess):
    def __init__(
        self,
        encoder: TELBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
