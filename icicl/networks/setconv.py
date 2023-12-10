import math
from typing import List, Optional, Tuple

import torch
from check_shapes import check_shapes
from torch import nn


class SetConvEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        points_per_unit: int,
        init_lengthscale: float,
        margin: float,
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
        return 1e-5 + nn.functional.softplus(self.lengthscale_param)

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ...]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
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
        init_lengthscale: float,
        scaling_factor: float,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        init_lengthscale = torch.as_tensor(dim * [init_lengthscale], dtype=dtype)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=train_lengthscale,
        )
        self.scaling_factor = scaling_factor

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(self.lengthscale_param)

    @check_shapes(
        "grids[0]: [m, ..., dx]",
        "grids[1]: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
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
        )  # shape (batch_size, num_trg, num_grid_points)

        z_grid = (weights @ z_grid) / self.scaling_factor

        return z_grid  # shape (batch_size, num_trg, Dz)


def make_adaptive_grid(
    x: torch.Tensor,
    points_per_unit: int,
    margin: float,
) -> torch.Tensor:
    """Create grids

    Arguments:
        x: Tensor of shape (batch_size, num_points, dim) containing the
            points.
        points_per_unit: Number of points per unit length in each dimension.
        margin: Margin around the points in `x`.

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """

    # Compute the lower and upper corners of the box containing the points
    xmin = torch.min(x, dim=-2)[0]
    xmax = torch.max(x, dim=-2)[0]

    return make_grid(
        xmin=xmin,
        xmax=xmax,
        points_per_unit=points_per_unit,
        margin=margin,
    )


def make_grid(
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    points_per_unit: int,
    margin: float,
) -> torch.Tensor:
    """Create grids

    Arguments:
        xmin: Tensor of shape (batch_size, dim) containing the lower
            corner of the box.
        xmax: Tensor of shape (batch_size, dim) containing the upper
            corner of the box.
        points_per_unit: Number of points per unit length in each dimension.
        margin: Margin around the box.

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """

    # Get grid dimension
    dim = xmin.shape[-1]

    # Compute half the number of points in each dimension
    num_points = torch.ceil(
        (0.5 * (xmax - xmin) + margin) * points_per_unit
    )  # shape (batch_size, dim)

    # Take the maximum over the batch, in order to use the same number of
    # points across all tasks in the batch, to enable tensor batching
    num_points = torch.max(num_points, dim=0)[0]
    num_points = 2 ** torch.ceil(torch.log(num_points) / math.log(2.0))  # shape (dim,)

    # Compute midpoints of each dimension, multiply integer grid by the grid
    # spacing and add midpoint to obtain dimension-wise grids
    x_mid = 0.5 * (xmin + xmax)  # shape (batch_size, dim)

    # Compute multi-dimensional grid
    grid = torch.stack(
        torch.meshgrid(
            *[
                torch.range(-num_points[i], num_points[i], dtype=xmin.dtype)
                for i in range(dim)
            ]
        ),
        axis=-1,
    ).to(
        x_mid
    )  # shape (n1, n2, ..., ndim, dim)

    for _ in range(dim):
        x_mid = torch.unsqueeze(x_mid, axis=-2)

    # Multiply integer grid by the grid spacing and add midpoint
    grid = x_mid + grid[None, ...] / points_per_unit

    return grid


def flatten_grid(grid: torch.Tensor) -> torch.Tensor:
    """Flatten the grid tensor to a tensor of shape
    (batch_size, num_grid_points, dim).

    Arguments:
        grid: Tensor of shape (batch_size, n1, n2, ..., ndim, dim)

    Returns:
        Tensor of shape (batch_size, num_grid_points, dim)
    """
    return torch.reshape(grid, shape=(grid.shape[0], -1, grid.shape[-1]))


def compute_eq_weights(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
) -> torch.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2)
    """

    # Expand dimensions for broadcasting
    x1 = x1[:, :, None, :]
    x2 = x2[:, None, :, :]
    lengthscales = lengthscales[None, None, None, :]

    # Compute pairwise distances between x1 and x2
    dist2 = torch.sum(
        ((x1 - x2) / lengthscales).pow(2),
        dim=-1,
    )  # shape (batch_size, num_x1, num_x2)

    # Compute weights
    weights = torch.exp(-0.5 * dist2)  # shape (batch_size, num_x1, num_x2)

    return weights
