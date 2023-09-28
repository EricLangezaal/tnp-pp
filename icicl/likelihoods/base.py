from abc import ABC, abstractmethod

import torch
import torch.distributions as td
from torch import nn


class Likelihood(nn.Module, ABC):
    out_dim_multiplier = 1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> td.Distribution:
        raise NotImplementedError
