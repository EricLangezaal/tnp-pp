"""
Implements linear model generation from https://arxiv.org/pdf/2306.15063.pdf
"""

from typing import Optional, Tuple

import gpytorch
import torch

from .data import SyntheticGenerator
from .gp import GPGroundTruthPredictor


class LinearGenerator(SyntheticGenerator):
    def __init__(self, *, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
        xic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, GPGroundTruthPredictor, Optional[torch.Tensor]]:
        w = self.sample_weight()

        # Generate observations.
        f = x @ w[..., None]
        y = f + self.noise_std * torch.randn_like(f)

        gt_pred = LinearGroundTruthPredictor(prior_std=1.0, noise_std=self.noise_std)

        return y, gt_pred, None

    def sample_weight(self) -> torch.Tensor:
        # Sample weight vector.
        w = torch.randn((self.batch_size, self.dim))

        return w


class LinearGroundTruthPredictor(GPGroundTruthPredictor):
    def __init__(self, prior_std: float, noise_std: float):
        kernel = gpytorch.kernels.LinearKernel()
        kernel.variance = prior_std**2

        super().__init__(kernel, noise_std)
