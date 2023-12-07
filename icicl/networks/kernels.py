from abc import ABC

import torch
from check_shapes import check_shapes
from torch import nn

from .mlp import MLP


class Kernel(nn.Module, ABC):
    pass


class RBFKernel(Kernel):
    def __init__(self, in_dim: int, out_dim: int = 1, init_lengthscale: float = 0.1):
        super().__init__()

        init_lengthscale = torch.as_tensor(in_dim * [out_dim * [init_lengthscale]])
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log()
        )

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

    @check_shapes("diff: [m, n1, n2, dx]", "return: [m, n1, n2, dout]")
    def forward(
        self,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        lengthscale = self.lengthscale[None, None, None, ...]
        diff = diff[..., None]

        # (m, n1, n2, h).
        dist = (diff / lengthscale).sum(-2, keepdim=False)
        dots = -0.5 * dist**2.0

        return dots


class MLPKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__()

        self.mlp = MLP(**kwargs)

    @check_shapes("diff: [m, n1, n2, dx]", "return: [m, n1, n2, dout]")
    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        dots = self.mlp(diff)

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

    @check_shapes("diff: [m, n1, n2, dx]", "return: [m, n1, n2, dout]")
    def foward(
        self,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        dots = sum(
            weight * kernel(diff) for weight, kernel in zip(self.weights, self.kernels)
        )
        return dots
