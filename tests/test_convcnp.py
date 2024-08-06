import pytest
import torch

from tnp.likelihoods.gaussian import NormalLikelihood
from tnp.models.convcnp import ConvCNP, ConvCNPDecoder, ConvCNPEncoder
from tnp.networks.cnn import CNN, UNet
from tnp.networks.mlp import MLP
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

    encoder_resizer = MLP(
        in_dim=2, out_dim=num_channels, num_layers=2, width=num_channels
    )

    convcnp_encoder = ConvCNPEncoder(
        conv_net=unet, setconv_encoder=setconv_encoder, resizer=encoder_resizer
    )

    decoder_resizer = MLP(
        in_dim=num_channels,
        out_dim=1,
        num_layers=2,
        width=num_channels,
    )

    convcnp_decoder = ConvCNPDecoder(
        setconv_decoder=setconv_decoder,
        resizer=decoder_resizer,
    )

    likelihood = NormalLikelihood(noise=1e-1)

    convcnp = ConvCNP(
        convcnp_encoder,
        convcnp_decoder,
        likelihood,
    )

    convcnp(xc, yc, xt)


@pytest.mark.parametrize("ndim", [1, 2])
def test_setconv_cnn(ndim: int):
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

    encoder_resizer = MLP(
        in_dim=2, out_dim=num_channels, num_layers=2, width=num_channels
    )

    convcnp_encoder = ConvCNPEncoder(
        conv_net=unet, setconv_encoder=setconv_encoder, resizer=encoder_resizer
    )

    decoder_resizer = MLP(
        in_dim=num_channels,
        out_dim=1,
        num_layers=2,
        width=num_channels,
    )

    convcnp_decoder = ConvCNPDecoder(
        setconv_decoder=setconv_decoder,
        resizer=decoder_resizer,
    )

    likelihood = NormalLikelihood(noise=1e-1)

    convcnp = ConvCNP(
        convcnp_encoder,
        convcnp_decoder,
        likelihood,
    )

    convcnp(xc, yc, xt)
