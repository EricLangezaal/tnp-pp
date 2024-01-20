from typing import List, Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.tetransformer import TETNPDTransformerEncoder, TETransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import NeuralProcess
from .tnp import TNPDDecoder, gen_tnpd_mask


class TETNPDEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TETransformerEncoder,
        y_encoder: nn.Module,
        initial_token_x_dependencies: Optional[List[int]] = None,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.y_encoder = y_encoder
        self.inital_token_x_dependencies = initial_token_x_dependencies

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=-2)
        y = torch.cat((yc, yt), dim=-2)

        if self.initial_token_dependencies is not None:
            z = self.y_encoder(
                torch.cat((y, x[..., self.initial_token_dependencies]), dim=-1)
            )
        else:
            z = self.y_encoder(y)

        # Construct mask.
        mask = gen_tnpd_mask(xc, xt, targets_self_attend=True)

        z = self.transformer_encoder(z, x, mask)
        return z


class EfficientTETNPDEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TETNPDTransformerEncoder,
        y_encoder: nn.Module,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = self.y_encoder(yc)
        zt = self.y_encoder(yt)

        zt = self.transformer_encoder(zc, zt, xc, xt)
        return zt


class TETNPD(NeuralProcess):
    def __init__(
        self,
        encoder: Union[TETNPDEncoder, EfficientTETNPDEncoder],
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
