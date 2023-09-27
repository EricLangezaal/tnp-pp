from typing import List, Tuple

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
        xy_comb=lambda x, y: torch.cat((x, y), dim=-1),
        agg=lambda x: torch.sum(x, dim=-2),
    ):
        super().__init__()

        self.z_encoder = z_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.xy_comb = xy_comb
        self.agg = agg

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


class DatasetDeepSet(nn.Module):
    def __init__(
        self,
        deepset: nn.Module,
        agg=lambda x: torch.sum(x, dim=-2),
    ):
        super().__init__()
        self.deepset = deepset
        self.agg = agg

    @check_shapes(
        "d[all][0]: [., ., d]",
        "d[all][1]: [., ., p]",
    )
    def forward(self, d: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.agg(torch.stack([self.deepset(d_[0], d_[1]) for d_ in d]))


class ICDeepSet(nn.Module):
    def __init__(
        self,
        deepset: nn.Module,
        dataset_deepset: nn.Module,
        z_comb=lambda z1, z2: torch.cat((z1, z2), dim=-1),
    ):
        super().__init__()

        self.deepset = deepset
        self.dataset_deepset = dataset_deepset
        self.z_comb = z_comb

    @check_shapes(
        "x: [m, n, dx]",
        "y: [m, n, dy]",
        "d[all][all][0]: [., dx]",
        "d[all][all][1]: [., dy]",
    )
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        d: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    ):
        assert len(d) == len(x), "Batch sizes do not match."
        return self.z_comb(self.deepset(x, y), self.dataset_deepset(d))
