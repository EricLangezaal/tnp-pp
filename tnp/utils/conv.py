import torch
from torch import nn
import math
from typing import Tuple, Optional, Callable

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
    
def compute_eq_weights(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
    dist_func: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,) or (dim, num_lengthscales)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2) or (batch_size, num_x1, num_x2, num_lengthscales)
    """
    if dist_func is None:
        dist_func = lambda x1, x2: x1 - x2

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
            (dist_func(x1, x2) / lengthscales).pow(2),
            dim=-2,
        )  # shape (batch_size, num_x1, num_x2, num_lengthscales)
    elif len(lengthscales.shape) == 4:
        dist2 = torch.sum(
            (dist_func(x1, x2) / lengthscales).pow(2),
            dim=-1,
        )  # shape (batch_size, num_x1, num_x2)
    else:
        raise ValueError("Invalid shape for `lengthscales`.")

    # Compute weights
    weights = torch.exp(-0.5 * dist2)
    return weights


def haversine_dist(
    x1: torch.Tensor, x2: torch.Tensor, latlon_dims: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    """
    Taken from https://www.movable-type.co.uk/scripts/latlong.html
    Setting R=1
    """
    lat1, lon1 = x1[..., latlon_dims[0], None], x1[..., latlon_dims[1], None]
    lat2, lon2 = x2[..., latlon_dims[0], None], x2[..., latlon_dims[1], None]

    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2).pow(2) + torch.cos(lat1) * torch.cos(lat2) * torch.sin(
        dlon / 2
    ).pow(2)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return c
