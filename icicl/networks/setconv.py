import math
from typing import List, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.conv import flatten_grid, make_adaptive_grid, make_grid, compute_eq_weights


class SetConvEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        init_lengthscale: float,
        margin: float,
        points_per_unit: int,
        xmin: Optional[List[float]] = None,
        xmax: Optional[List[float]] = None,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        init_lengthscale = torch.as_tensor(dim * [init_lengthscale], dtype=dtype)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=train_lengthscale,
        )
        self.points_per_unit = points_per_unit
        self.margin = margin
        self.xmin = torch.as_tensor(xmin) if xmin is not None else None
        self.xmax = torch.as_tensor(xmax) if xmax is not None else None

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        yc = torch.cat((yc, torch.ones(*yc.shape[:-1], 1).to(yc)), dim=-1)

        # Build dimension wise grids.
        if self.xmin is None or self.xmax is None:
            x_grid = make_adaptive_grid(
                x=torch.concat((xc, xt), dim=-2),
                points_per_unit=self.points_per_unit,
                margin=self.margin,
            )
        else:
            x_grid = make_grid(
                xmin=self.xmin,
                xmax=self.xmax,
                points_per_unit=self.points_per_unit,
                margin=self.margin,
            )

        # Shape (batch_size, num_grid_points, dx).
        x_grid_flat = flatten_grid(x_grid)

        # Compute matrix of weights between context points and grid points.
        # (batch_size, nc, num_grid_points).
        weights = compute_eq_weights(
            x1=x_grid_flat, x2=xc, lengthscales=self.lengthscale
        )

        # Multiply context outputs by weights.
        # (batch_size, num_grid_points, 2).
        z_grid_flat = weights @ yc

        # Reshape grid.
        # (batch_size, n1, ..., ndim, 2).
        z_grid = torch.reshape(
            z_grid_flat,
            shape=x_grid.shape[:-1] + z_grid_flat.shape[-1:],
        )

        return x_grid, z_grid


class SetConvDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        init_lengthscale: float = 0.1,
        scaling_factor: float = 1.0,
        num_kernels: int = 1,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Compute log-spacing around init_lengthscale, so max_init_lengthscale = 10 * min_init_lengthscale.
        log_init_lengthscale = math.log(init_lengthscale, 10)
        min_log_init_lengthscale = log_init_lengthscale - 0.5
        max_log_init_lengthscale = log_init_lengthscale + 0.5
        init_lengthscales = torch.logspace(
            min_log_init_lengthscale,
            max_log_init_lengthscale,
            steps=num_kernels,
            dtype=dtype,
        )
        init_lengthscales = einops.repeat(init_lengthscales, "nk -> d nk", d=dim)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscales).exp() - 1).log(),
            requires_grad=train_lengthscale,
        )
        self.scaling_factor = scaling_factor

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

    @check_shapes(
        "grids[0]: [m, ..., dx]",
        "grids[1]: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dout]",
    )
    def forward(
        self, grids: Tuple[torch.Tensor, torch.Tensor], xt: torch.Tensor
    ) -> torch.Tensor:
        """Apply EQ kernel smoothing to the grid points,
        to interpolate to the target points.

        Arguments:
            x_grid: Tensor of shape (batch_size, n1, ..., ndim, Dx)
            z_grid: Tensor of shape (batch_size, n1, ..., ndim, Dz)
            x_trg: Tensor of shape (batch_size, num_trg, Dx)

        Returns:
            Tensor of shape (batch_size, num_trg, dim)
        """
        x_grid, z_grid = grids

        # Flatten grids
        x_grid = flatten_grid(x_grid)  # shape (batch_size, num_grid_points, Dx)
        z_grid = flatten_grid(z_grid)  # shape (batch_size, num_grid_points, Dz)

        # Compute weights
        weights = compute_eq_weights(
            x1=xt,
            x2=x_grid,
            lengthscales=self.lengthscale,
        )  # shape (batch_size, num_trg, num_grid_points, num_kernels)

        # Shape (batch_size, num_kernels, num_trg, num_grid_points).
        weights = einops.rearrange(weights, "b nt ng nk -> b nk nt ng")

        # Shape (batch_size, num_kernels, num_trg, z_dim).
        z_grid = (weights @ z_grid[:, None, ...]) / self.scaling_factor
        z_grid = einops.rearrange(z_grid, "b nk nt dz -> b nt (dz nk)")

        return z_grid  # shape (batch_size, num_trg, Dz x num_kernels)
