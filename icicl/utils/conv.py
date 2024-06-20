import torch
from torch import nn
import math
from typing import Tuple, Optional

from .initialisation import weights_init


def make_depth_sep_conv(Conv: nn.Module):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            bias: bool = True,
            **kwargs
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x: torch.Tensor):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


# @check_shapes("x1: [m, c1, n1]", "x2: [m, c2, n2]", "out: [m, c1+c2, n]")
def pad_concat(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Concat the activations of two layer channel-wise by padding the layer with fewer points with zeros.

    Args:
        x1 (torch.Tensor): Activations from first layers.
        x2 (torch.Tensor): activations from second layers.

    Returns:
        torch.Tensor: Concatenated activations of both layers of shape.
    """
    if x1.shape[2] > x2.shape[2]:
        padding = x1.shape[2] - x2.shape[2]
        if padding % 2 == 0:  # Even difference
            x2 = nn.functional.pad(  # pylint: disable=not-callable
                x2, (int(padding / 2), int(padding / 2)), "reflect"
            )
        else:  # Odd difference
            x2 = nn.functional.pad(  # pylint: disable=not-callable
                x2, (int((padding - 1) / 2), int((padding + 1) / 2)), "reflect"
            )
    elif x2.shape[2] > x1.shape[2]:
        padding = x2.shape[2] - x1.shape[2]
        if padding % 2 == 0:  # Even difference
            x1 = nn.functional.pad(  # pylint: disable=not-callable
                x1, (int(padding / 2), int(padding / 2)), "reflect"
            )
        else:  # Odd difference
            x1 = nn.functional.pad(  # pylint: disable=not-callable
                x1, (int((padding - 1) / 2), int((padding + 1) / 2)), "reflect"
            )

    return torch.cat([x1, x2], dim=1)


def to_multiple(x: torch.Tensor, multiple: int):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    return x + multiple - x % multiple


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


def convNd(n: int, **kwargs) -> nn.Module:
    try:
        return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[n - 1](**kwargs)
    except:
        raise NotImplementedError
    

def compute_eq_weights(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
) -> torch.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,) or (dim, num_lengthscales)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2) or (batch_size, num_x1, num_x2, num_lengthscales)
    """

    # Expand dimensions for broadcasting
    x1 = x1[:, :, None, :]
    x2 = x2[:, None, :, :]
    lengthscales = lengthscales[None, None, None, ...]

    # Compute pairwise distances between x1 and x2
    # TODO: Kinda hate this.
    if len(lengthscales.shape) == 5:
        x1 = x1[..., None]
        x2 = x2[..., None]
        dist2 = torch.sum(
            ((x1 - x2) / lengthscales).pow(2),
            dim=-2,
        )  # shape (batch_size, num_x1, num_x2, num_lengthscales)
    elif len(lengthscales.shape) == 4:
        dist2 = torch.sum(
            ((x1 - x2) / lengthscales).pow(2),
            dim=-1,
        )  # shape (batch_size, num_x1, num_x2)
    else:
        raise ValueError("Invalid shape for `lengthscales`.")

    # Compute weights
    weights = torch.exp(-0.5 * dist2)

    return weights