import pytest
import torch

from tnp.networks.setconv import SetConvDecoder, SetConvEncoder


@pytest.mark.parametrize("ndim", [1, 2])
def test_setconv(ndim: int):
    # Set test constants
    nc = 128
    nt = 64
    m = 16
    points_per_unit = 10
    init_lengthscale = 0.5
    scaling_factor = 1.0
    margin = 0.1

    # Dimensionwise grid coordinates.
    xc = torch.randn(m, nc, ndim)
    yc = torch.randn(m, nc, 1)
    xt = torch.randn(m, nt, ndim)

    setconv_encoder = SetConvEncoder(
        dim=ndim,
        points_per_unit=points_per_unit,
        init_lengthscale=init_lengthscale,
        margin=margin,
    )

    setconv_decoder = SetConvDecoder(
        dim=ndim,
        init_lengthscale=init_lengthscale,
        scaling_factor=scaling_factor,
    )

    x_grid, z_grid = setconv_encoder(xc, yc, xt)
    z_grid = setconv_decoder((x_grid, z_grid), xt)
