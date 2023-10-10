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
