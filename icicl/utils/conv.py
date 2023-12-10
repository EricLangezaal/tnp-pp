import torch
from torch import nn

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
