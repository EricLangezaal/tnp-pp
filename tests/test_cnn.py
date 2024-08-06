import pytest
import torch

from tnp.networks.cnn import CNN


@pytest.mark.parametrize("ndim", [1, 2])
def test_cnn(ndim: int):
    # Set test constants
    m = 16
    dy = 1
    points_per_dim = 100
    z_grid = torch.randn(m, *[points_per_dim] * ndim, dy + 1)

    num_channels = dy + 1
    num_blocks = 5

    cnn = CNN(
        dim=ndim,
        num_channels=num_channels,
        num_blocks=num_blocks,
    )
    z_grid = cnn(z_grid)

    assert z_grid.shape == torch.Size((m, *[points_per_dim] * ndim, dy + 1))
