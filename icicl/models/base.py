from typing import List, Tuple

import torch
from check_shapes import check_shapes
from torch import nn


class NeuralProcess(nn.Module):
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

    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))


class ICNeuralProcess(nn.Module):
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

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "dc[all][all][0]: [., dx]",
        "dc[all][all][1]: [., dy]",
        "return: [m, nt, .]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        dc: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    ):
        assert len(dc) == len(xc), "Batch sizes do not match."
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, dc), xt))
