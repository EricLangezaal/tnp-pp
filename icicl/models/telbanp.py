from typing import List, Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.tetransformer import (
    NestedTEISetTransformerEncoder,
    NestedTEPerceiverEncoder,
)
from ..utils.helpers import preprocess_observations
from .base import NeuralProcess
from .lbanp import LBANPDecoder


class TELBANPEncoder(nn.Module):
    def __init__(
        self,
        nested_perceiver_encoder: Union[
            NestedTEPerceiverEncoder, NestedTEISetTransformerEncoder
        ],
        y_encoder: nn.Module,
        initial_token_x_dependencies: Optional[List[int]] = None,
    ):
        super().__init__()

        self.nested_perceiver_encoder = nested_perceiver_encoder
        self.y_encoder = y_encoder
        self.inital_token_x_dependencies = initial_token_x_dependencies

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

        if self.initial_token_dependencies is not None:
            zc = self.y_encoder(
                torch.cat((yc, xc[..., self.initial_token_dependencies]), dim=-1)
            )
            zt = self.y_encoder(
                torch.cat((yt, xt[..., self.initial_token_dependencies]), dim=-1)
            )
        else:
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
