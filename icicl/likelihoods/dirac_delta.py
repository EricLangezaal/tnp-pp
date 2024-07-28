import torch
import gpytorch.distributions as dist

from .base import Likelihood

class DeltaLikelihood(Likelihood):

    def forward(self, x: torch.Tensor) -> dist.delta.Delta:
        return dist.delta.Delta(x)