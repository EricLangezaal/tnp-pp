from typing import Callable

import torch
from check_shapes import check_shapes
from torch import nn


class DeepSet(nn.Module):
    """Deep set.

    Args:
        phi (object): Pre-aggregation function.
        agg (object, optional): Aggregation function. Defaults to summing.

    Attributes:
        phi (object): Pre-aggregation function.
        agg (object): Aggregation function.
    """

    def __init__(
        self,
        z_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        xy_comb: Callable = lambda x, y: torch.cat((x, y), dim=-1),
        agg: str = "sum",
    ):
        super().__init__()

        self.z_encoder = z_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.xy_comb = xy_comb

        if agg == "sum":
            self.agg = lambda x: torch.sum(x, dim=-2)
        elif agg == "mean":
            self.agg = lambda x: torch.mean(x, dim=-2)
        else:
            raise ValueError("agg must be one of 'sum', 'mean'")

    @check_shapes(
        "x: [m, n, dx]",
        "y: [m, n, dy]",
        "return: [m, dz]",
    )
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_enc = self.x_encoder(x)
        y_enc = self.y_encoder(y)
        z = self.xy_comb(x_enc, y_enc)
        z = self.z_encoder(z)
        z = self.agg(z)
        return z
