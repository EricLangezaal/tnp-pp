from abc import ABC
from typing import Callable

import gpytorch
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


class SpatialGeneralisationMLPKernel(MLPKernel):
    def __init__(self, dim_x: int, lengthscale: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.dim_x = dim_x
        self.lengthscale = torch.as_tensor([lengthscale] * dim_x)

    @check_shapes("diff: [m, n1, n2, dx]", "return: [m, n1, n2, dout]")
    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        dots = self.mlp(diff)

        # Get actual x_difference.
        diff_x = diff[..., -self.dim_x :]

        # (m, n1, n2, h).
        dist_x = (diff_x / self.lengthscale).sum(-1, keepdim=True)
        dots_x = -0.5 * dist_x**2.0

        return dots_x.exp() * dots


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


class GibbsKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        lengthscale_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.where(
            x[..., 0][..., None] < 0,
            torch.ones(*x[..., 0][..., None].shape).to(x) * 1.0,
            torch.ones(*x[..., 0][..., None].shape).to(x) * 0.25,
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lengthscale_fn = lengthscale_fn

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        x1_lengthscale = self.lengthscale_fn(x1)
        x2_lengthscale = self.lengthscale_fn(x2)
        lengthscale = (x1_lengthscale**2 + x2_lengthscale**2) ** 0.5
        const = ((2 * x1_lengthscale * x2_lengthscale) / lengthscale**2) ** 0.5

        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or last_dim_is_batch
            or gpytorch.settings.trace_mode.on()
        ):
            x1_ = x1.div(lengthscale)
            x2_ = x2.div(lengthscale)
            return const * self.covar_dist(
                x1_,
                x2_,
                square_dist=True,
                diag=diag,
                dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf,
                postprocess=True,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )
        return const * gpytorch.functions.RBFCovariance.apply(
            x1,
            x2,
            lengthscale,
            lambda x1, x2: self.covar_dist(
                x1,
                x2,
                square_dist=True,
                diag=False,
                dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf,
                postprocess=False,
                last_dim_is_batch=False,
                **params,
            ),
        )
