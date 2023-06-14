from typing import List, Optional, Tuple

import torch
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
        phi,
        agg=lambda x: torch.sum(x, dim=-2),
    ):
        super().__init__()
        self.phi = phi
        self.agg = agg

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.cat((x, y), dim=-1)
        z = self.phi(z)
        z = self.agg(z)  # Aggregates over the data dimension.

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

    def forward(self, d: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        return self.agg(torch.stack([self.deepset(d_[0], d_[1]) for d_ in d]))


class ICDeepSet(nn.Module):
    def __init__(
        self,
        deepset: nn.Module,
        dataset_deepset: nn.Module,
        agg: lambda z1, z2: torch.cat((z1, z2), dim=-1),
    ):
        super().__init__()
        self.deepset = deepset
        self.dataset_deepset = dataset_deepset
        self.agg = agg

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        d: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        return self.agg(self.deepset(x, y), self.dataset_deepset(d))
