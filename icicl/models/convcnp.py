import torch
from torch import nn

from ..networks.setconv import SetConvDecoder, SetConvEncoder
from .base import NeuralProcess


class ConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        setconv_encoder: SetConvEncoder,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.setconv_encoder = setconv_encoder

    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        x_grid, z_grid = self.setconv_encoder(xc, yc, xt)
        z_grid = self.conv_net(z_grid)

        return x_grid, z_grid


class ConvCNP(NeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: SetConvDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
