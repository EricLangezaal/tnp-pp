import pytest
import torch

from icicl.likelihoods.gaussian import NormalLikelihood
from icicl.models.convcnp import ConvCNP, ConvCNPEncoder
from icicl.networks.cnn import CNN, UNet
from icicl.networks.setconv import SetConvDecoder, SetConvEncoder

# @pytest.mark.parametrize("ndim", [1, 2])
# def test_setconv(ndim: int):
#     # Set test constants
#     nc = 128
#     nt = 64
#     m = 16
#     dy = 1
#     points_per_unit = 10
#     init_lengthscale = 0.5
#     scaling_factor = 1.0
#     margin = 0.1

#     # UNet params.
#     in_channels = dy + 1
#     first_channels = 16
#     last_channels = dy
#     kernel_size = 3
#     num_channels = [32, 64, 32, 16]

#     # Dimensionwise grid coordinates.
#     xc = torch.randn(m, nc, ndim)
#     yc = torch.randn(m, nc, 1)
#     xt = torch.randn(m, nt, ndim)

#     setconv_encoder = SetConvEncoder(
#         dim=ndim,
#         points_per_unit=points_per_unit,
#         init_lengthscale=init_lengthscale,
#         margin=margin,
#     )

#     setconv_decoder = SetConvDecoder(
#         dim=ndim,
#         init_lengthscale=init_lengthscale,
#         scaling_factor=scaling_factor,
#     )

#     unet = UNet(
#         dim=ndim,
#         in_channels=in_channels,
#         first_channels=first_channels,
#         last_channels=last_channels,
#         kernel_size=kernel_size,
#         num_channels=num_channels,
#     )

#     convcnp_encoder = ConvCNPEncoder(unet, setconv_encoder)

#     likelihood = NormalLikelihood(noise=1e-1)

#     convcnp = ConvCNP(
#         convcnp_encoder,
#         setconv_decoder,
#         likelihood,
#     )

#     convcnp(xc, yc, xt)


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

    # UNet params.
    num_channels = 2
    num_blocks = 5

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

    unet = UNet(dim=ndim, num_channels=num_channels, num_blocks=num_blocks)

    convcnp_encoder = ConvCNPEncoder(unet, setconv_encoder)

    likelihood = NormalLikelihood(noise=1e-1)

    convcnp = ConvCNP(
        convcnp_encoder,
        setconv_decoder,
        likelihood,
    )

    convcnp(xc, yc, xt)


@pytest.mark.parametrize("ndim", [1, 2])
def test_setcon_cnn(ndim: int):
    # Set test constants
    nc = 128
    nt = 64
    m = 16
    points_per_unit = 10
    init_lengthscale = 0.5
    scaling_factor = 1.0
    margin = 0.1

    # UNet params.
    num_channels = 2
    num_blocks = 5

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

    unet = CNN(dim=ndim, num_channels=num_channels, num_blocks=num_blocks)

    convcnp_encoder = ConvCNPEncoder(unet, setconv_encoder)

    likelihood = NormalLikelihood(noise=1e-1)

    convcnp = ConvCNP(
        convcnp_encoder,
        setconv_decoder,
        likelihood,
    )

    convcnp(xc, yc, xt)
