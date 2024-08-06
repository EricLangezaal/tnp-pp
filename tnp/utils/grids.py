from typing import Tuple, Optional
import itertools
import math

from check_shapes import check_shapes 
import torch
from torch import nn
import torch.nn.functional as F

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
    """Create grids, where number of points is always power of 2.

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
        batch_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    
    grid_range = torch.as_tensor(grid_range)
    grid = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(
                    grid_range[i, 0],
                    grid_range[i, 1],
                    steps=int((grid_range[i, 1] - grid_range[i, 0]) * points_per_unit),
                    dtype=torch.float,
                )
                for i in range(len(grid_range))
            ],
            indexing='ij',
        ),
        dim=-1,
    )
    if batch_shape is not None:
        grid = grid.expand(*batch_shape, *grid.shape)
    return grid


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
        grid_shape: Optional[Tuple[int]] = None
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


@check_shapes(
    "x: [m, n, dx]",
    "x_grid: [m, ..., dx]",
    "return[0]: [m, n, k]",
    "return[1]: [m, n, k]",
)
def nearest_gridded_neighbours(
    x: torch.Tensor,
    x_grid: torch.Tensor,
    k: int = 1,
    roll_dims: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_shape = torch.as_tensor(x_grid.shape[1:-1], device=x.device)
    x_grid_flat = flatten_grid(x_grid)

    # Get number of neighbors along each dimension.
    dim_x = x.shape[-1]
    num_grid_spacings = math.ceil(k ** (1 / dim_x))
    
    # Set roll_dims to the actual index if they are specified as (-x, )
    num_dims = len(grid_shape)
    if roll_dims is not None:
        roll_dims = tuple(roll_dim % num_dims for roll_dim in roll_dims)

    # Quick calculation of nearest grid neighbour.
    x_grid_min = x_grid_flat.amin(dim=(0, 1))
    x_grid_max = x_grid_flat.amax(dim=(0, 1))
    x_grid_spacing = (x_grid_max - x_grid_min) / (grid_shape - 1)

    nearest_multi_idx = (x - x_grid_min + x_grid_spacing / 2) // x_grid_spacing

    # Generate a base grid for combinations of grid_spacing offsets from main neighbor.
    base_grid = torch.tensor(
        list(
            itertools.product(
                torch.arange(
                    -num_grid_spacings // 2 + num_grid_spacings % 2,
                    num_grid_spacings // 2 + 1,
                ),
                repeat=dim_x,
            )
        ),
        device=x.device,
    ).float()

    # Reshape and expand the base grid
    base_grid = base_grid.view(1, 1, -1, dim_x).expand(
        *nearest_multi_idx.shape[:-1], -1, -1
    )
    # Expand the indices of nearest neighbors to account for more than 1.
    nearest_multi_idx_expanded = nearest_multi_idx.unsqueeze(2).expand(
        -1, -1, (num_grid_spacings + 1 - num_grid_spacings % 2) ** dim_x, -1
    )
    # Generate all combinations by adding the offsets to the main neighbor.
    nearest_multi_idx = nearest_multi_idx_expanded + base_grid

    # If not rolling_dims, do not allow neighbors to go off-grid.
    # Otherwise, roll the grid along the specified dimension.
    if roll_dims is None:
        nearest_multi_idx = torch.max(
            torch.min(nearest_multi_idx, grid_shape - 1), torch.zeros_like(grid_shape)
        ).squeeze(-2)
    else:
        nearest_multi_idx = torch.cat(
            [
                (
                    torch.max(
                        torch.min(nearest_multi_idx[..., i], grid_shape[i] - 1),
                        torch.tensor(0),
                    ).unsqueeze(-1)
                    if (i not in roll_dims)
                    else (nearest_multi_idx[..., i] % grid_shape[i]).unsqueeze(-1)
                )
                for i in range(num_dims)
            ],
            dim=-1,
        ).squeeze(-2)

    # Get strides.
    strides = torch.flip(
        torch.cumprod(
            torch.cat((torch.ones((1,), device=grid_shape.device), grid_shape), dim=0),
            dim=0,
        )[:-1],
        dims=(0,),
    )

    # (batch_size, nt, num_neighbors).
    if k == 1:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        )
    else:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        ).squeeze(-1)

    if k != 1:
        # Get mask for MHCA.
        mask = torch.ones_like(nearest_idx, dtype=torch.bool)

        # Sort nearest_idx.
        sorted_nearest_idx, indices = torch.sort(nearest_idx, dim=-1, stable=True)

        # Find first occurence where consecutive elements are different.
        first_occurrence = torch.ones_like(sorted_nearest_idx, dtype=torch.bool)
        first_occurrence[..., 1:] = (
            sorted_nearest_idx[..., 1:] != sorted_nearest_idx[..., :-1]
        )

        # Back to the original shape.
        original_indices = torch.argsort(indices, dim=-1)
        mask = torch.gather(first_occurrence, dim=-1, index=original_indices)
    else:
        mask = None

    return nearest_idx, mask


def coarsen_grid(grid: torch.Tensor, coarsen_factors: Tuple[int, ...]) -> torch.Tensor:
    grid = grid.movedim(-1, 1) # move data dim to channel location

    coarse_grid = func_AvgPoolNd(
        n=grid.ndim - 2, 
        input=grid,
        kernel_size=tuple(coarsen_factors), 
        stride=tuple(coarsen_factors),
    )
    coarse_grid = coarse_grid.movedim(1, -1) # move embed dim back to the end
    return coarse_grid


def convNdModule(n: int, **kwargs) -> nn.Module:
    try:
        return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[n - 1](**kwargs)
    except:
        raise NotImplementedError
    
def func_convNd(n: int, **kwargs) -> torch.Tensor:
    try:
        return (F.conv1d, F.conv2d, F.conv3d)[n - 1](**kwargs)
    except:
        raise NotImplementedError
    
def func_AvgPoolNd(n: int, **kwargs) -> torch.Tensor:
    try:
        return (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[n - 1](**kwargs)
    except:
        raise NotImplementedError

