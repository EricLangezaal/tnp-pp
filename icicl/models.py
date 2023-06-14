from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class NP(nn.Module):
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

    def forward(
        self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(x_c, y_c), x_t))


class ICNP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        dataset_encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.dataset_encoder = dataset_encoder
        self.decoder = decoder
        self.likelihood = likelihood

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        d_c: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        return self.likelihood(self.decoder(self.encoder(x_c, y_c, d_c), x_t))
