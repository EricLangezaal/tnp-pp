from typing import Tuple

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.setconv import SetConvDecoder, SetConvEncoder
from .base import NeuralProcess


class ConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        setconv_encoder: SetConvEncoder,
        resizer: nn.Module,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.setconv_encoder = setconv_encoder
        self.resizer = resizer

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        x_grid, z_grid = self.setconv_encoder(xc, yc, xt)
        # Increase dimension.
        z_grid = self.resizer(z_grid)
        z_grid = self.conv_net(z_grid)

        return x_grid, z_grid


class ConvCNPDecoder(nn.Module):
    def __init__(
        self,
        setconv_decoder: SetConvDecoder,
        resizer: nn.Module,
    ):
        super().__init__()

        self.setconv_decoder = setconv_decoder
        self.resizer = resizer

    @check_shapes(
        "grids[0]: [m, ..., dx]",
        "grids[1]: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dy]",
    )
    def forward(
        self, grids: Tuple[torch.Tensor, torch.Tensor], xt: torch.Tensor
    ) -> torch.Tensor:
        z_grid = self.setconv_decoder(grids, xt)
        z_grid = self.resizer(z_grid)
        return z_grid


class ConvCNP(NeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: ConvCNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
