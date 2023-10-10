from abc import ABC

import torch
from check_shapes import check_shapes
from torch import nn


class BaseNeuralProcess(nn.Module, ABC):
    """Represents a neural process base class"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class NeuralProcess(BaseNeuralProcess):
    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))


class ICNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xic: [m, nic, ncic, dx]",
        "yic: [m, nic, ncic, dy]",
        "xt: [m, nt, dx]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xic: torch.Tensor,
        yic: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xic, yic, xt), xt))
