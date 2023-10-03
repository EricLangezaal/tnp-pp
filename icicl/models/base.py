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
