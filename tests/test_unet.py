import pytest
import torch

from icicl.networks.conv import UNet


@pytest.mark.parametrize("ndim", [1, 2])
def test_unet(ndim: int):
    # Set test constants
    m = 16
    dy = 1
    points_per_dim = 100
    z_grid = torch.randn(m, *[points_per_dim] * ndim, dy + 1)

    in_channels = dy + 1
    first_channels = 32
    last_channels = 2
    kernel_size = 3
    num_channels = (32, 32, 32, 32, 32)
    strides = (2, 2, 2, 2, 2)

    unet = UNet(
        dim=ndim,
        in_channels=in_channels,
        first_channels=first_channels,
        last_channels=last_channels,
        kernel_size=kernel_size,
        num_channels=num_channels,
        strides=strides,
    )
    z_grid = unet(z_grid)

    assert z_grid.shape == torch.Size((m, *[points_per_dim] * ndim, dy + 1))
