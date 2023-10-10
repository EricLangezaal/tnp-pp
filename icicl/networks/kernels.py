from abc import ABC
from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn

from .mlp import MLP


class Kernel(nn.Module, ABC):
    pass


class RBFKernel(Kernel):
    def __init__(self, dim: int, init_lengthscale: float):
        super().__init__()

        init_lengthscale = torch.as_tensor(dim * [init_lengthscale])
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log()
        )

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(self.lengthscale_param)

    @check_shapes("diff: [m, n1, n2, dx]", "mask: [m, n1, n2]", "return: [m, n1, n2]")
    def forward(
        self, diff: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        lengthscale = self.lengthscale[None, None, None, :]
        dist = (diff / lengthscale).sum(-1)
        dots = torch.exp(-0.5 * dist**2.0)

        if mask is not None:
            dots = torch.mask_fill(dots, mask, 0)

        return dots


class MLPKernel(Kernel):
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        self.mlp = MLP(in_dim=dim, out_dim=1, **kwargs)

    @check_shapes("diff: [m, n1, n2, dx]", "mask: [m, n1, n2]", "return: [m, n1, n2]")
    def forward(
        self, diff: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dots = self.mlp(diff).squeeze(-1)

        if mask is not None:
            dots = torch.masked_fill(dots, mask, -float("inf"))

        dots = dots.softmax(dim=-1)

        return dots


class MixtureKernel(Kernel):
    def __init__(self, *kernels: Kernel, train_weights: bool = True):
        super().__init__()

        self.kernels = nn.ModuleList(list(kernels))
        self.weights_param = nn.Parameter(
            torch.ones(len(self.kernels)), requires_grad=train_weights
        )

    @property
    def weights(self):
        return nn.functional.softmax(self.weights_param, dim=-1)

    @check_shapes("diff: [m, n1, n2, dx]", "mask: [m, n1, n2]", "return: [m, n1, n2]")
    def foward(
        self, diff: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dots = sum(
            weight * kernel(diff, mask)
            for weight, kernel in zip(self.weights, self.kernels)
        )
        return dots
