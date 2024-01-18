from typing import Tuple

import einops
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


class GriddedConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        resizer: nn.Module,
    ):
        super().__init__()
        self.conv_net = conv_net
        self.resizer = resizer

    def forward(self, mc: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.resizer(z_grid)
        z_grid = self.conv_net(z_grid)
        return z_grid


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
        "grids[1]: [m, ..., dz]",
        "xt: [m, ...]",
        "return: [m, nt, dy]",
    )
    def forward(
        self, grids: Tuple[torch.Tensor, torch.Tensor], xt: torch.Tensor
    ) -> torch.Tensor:
        zt = self.setconv_decoder(grids, xt)
        zt = self.resizer(zt)
        return zt


class GriddedConvCNPDecoder(nn.Module):
    def __init__(
        self,
        resizer: nn.Module,
    ):
        super().__init__()

        self.resizer = resizer

    @check_shapes(
        "z_grid: [m, ..., dz]",
        "mt: [m, ...]",
        "return: [m, nt, dy]",
    )
    def forward(self, z_grid: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        zt = self.resizer(zt)
        return zt


class ConvCNP(NeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: ConvCNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class GriddedConvCNP(nn.Module):
    def __init__(
        self,
        encoder: GriddedConvCNPEncoder,
        decoder: GriddedConvCNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    @check_shapes("mc: [m, ...]", "y: [m, ..., dy]", "mt: [m, ...]")
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(mc, y), mt))
