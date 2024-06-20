from typing import Tuple, Optional
import math

import torch

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

    # Take the maximum over the batch, in order to use the same numbe`r of
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
                torch.linspace(-num_points[i], num_points[i], steps=num_points[0].to(int) * 2, dtype=xmin.dtype)
                for i in range(dim)
            ],
            indexing='ij'
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


def make_grid_from_range(
        grid_range: Tuple[Tuple[float, float], ...], 
        points_per_unit: int,
        batch_shape: Optional[torch.Size] = (1,),
) -> torch.Tensor:
    
    grid_range = torch.as_tensor(grid_range, dtype=torch.float)
    return make_grid(
            xmin = grid_range[:, 0].repeat(*batch_shape, 1), 
            xmax = grid_range[:, 1].repeat(*batch_shape, 1), 
            points_per_unit = points_per_unit, 
            margin = 0
    )


def flatten_grid(
        grid: torch.Tensor
) -> torch.Tensor:
    """Flatten the grid tensor to a tensor of shape
    (batch_size, num_grid_points, dim).

    Arguments:
        grid: Tensor of shape (batch_size, n1, n2, ..., ndim, dim)

    Returns:
        Tensor of shape (batch_size, num_grid_points, dim)
    """
    return torch.reshape(grid, shape=(grid.shape[0], -1, grid.shape[-1]))


def unflatten_grid(
        data: torch.Tensor, 
        grid_shape: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Return grid to its original shape:
    (batch_size, n1, n2, ..., ndim, dim)

    Arguments:
        grid: Tensor of shape (batch_size, num_grid_points, dim)

    Returns:
        Tensor of shape (batch_size, n1, n2, ..., ndim, dim)
    """
    if grid_shape is None:
        num_points = int(data.shape[-2] ** (1 / data.shape[-1]))
        grid_shape = (num_points, ) * data.shape[-1]
    
    return data.reshape(data.shape[:-2] + grid_shape + data.shape[-1:])