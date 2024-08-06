import torch
import torch.distributions as td
from torch import nn

from .base import Likelihood


class NormalLikelihood(Likelihood):
    def __init__(self, noise: float, train_noise: bool = True):
        super().__init__()

        self.log_noise = nn.Parameter(
            torch.as_tensor(noise).log(), requires_grad=train_noise
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    @noise.setter
    def noise(self, value: float):
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())

    def forward(self, x: torch.Tensor) -> td.Normal:
        return td.Normal(x, self.noise)


class HeteroscedasticNormalLikelihood(Likelihood):
    out_dim_multiplier = 2

    def __init__(self, min_noise: float = 0.0):
        super().__init__()

        self.min_noise = min_noise

    def forward(self, x: torch.Tensor) -> td.Normal:
        assert x.shape[-1] % 2 == 0

        loc, log_var = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        scale = nn.functional.softplus(log_var) ** 0.5 + self.min_noise
        return td.Normal(loc, scale)
