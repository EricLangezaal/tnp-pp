import torch
import torch.distributions as td
from torch import nn


class Likelihood(nn.Module):
    out_dim_multiplier = 1

    def forward(self, x: torch.Tensor) -> td.Distribution:
        raise NotImplementedError
