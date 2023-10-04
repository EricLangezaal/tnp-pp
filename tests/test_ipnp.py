import pytest
import torch

from icicl.likelihoods.gaussian import NormalLikelihood
from icicl.models.ipnp import IPNP, IPNPDecoder, IPNPEncoder
from icicl.networks.mlp import MLP
from icicl.networks.transformer import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
    SPINDecoder,
    SPINEncoder,
)


@pytest.mark.parametrize("ndim", [1, 2])
def test_ipnp(ndim: int):
    # Test parameters.
    m = 16
    nc = 128
    nt = 64
    num_latent_features = 3
    num_latent_datapoints = 8
    dy = 1

    # Encoder parameters.
    embed_dim = 16
    num_heads = 8
    head_dim = 3
    feedforward_dim = 16
    num_layers = 3

    xc = torch.randn(m, nc, ndim)
    yc = torch.randn(m, nc, 1)
    xt = torch.randn(m, nt, ndim)

    xaba_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    abla_layer = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    xabd_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim * num_latent_features,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    spin_encoder = SPINEncoder(
        num_latent_features=num_latent_features,
        num_latent_datapoints=num_latent_datapoints,
        xaba_layer=xaba_layer,
        abla_layer=abla_layer,
        xabd_layer=xabd_layer,
        num_layers=num_layers,
    )
    xaba_decoder_layer = xaba_layer
    abla_decoder_layer = abla_layer
    xabd_decoder_layer = xabd_layer
    spin_decoder = SPINDecoder(
        num_latent_features=num_latent_features,
        xaba_layer=xaba_decoder_layer,
        abla_layer=abla_decoder_layer,
        xabd_layer=xabd_decoder_layer,
        num_layers=1,
    )
    attribute_encoder = MLP(
        in_dim=1,
        out_dim=embed_dim,
        num_layers=2,
        width=feedforward_dim,
    )
    z_decoder = MLP(
        in_dim=embed_dim * num_latent_features,
        out_dim=dy + 1,
        num_layers=2,
        width=feedforward_dim,
    )

    ipnp_encoder = IPNPEncoder(
        spin_encoder=spin_encoder,
        attribute_encoder=attribute_encoder,
    )
    ipnp_decoder = IPNPDecoder(
        spin_decoder=spin_decoder,
        attribute_encoder=attribute_encoder,
        z_decoder=z_decoder,
        dy=dy,
    )
    likelihood = NormalLikelihood(noise=1e-1)

    ipnp = IPNP(ipnp_encoder, ipnp_decoder, likelihood=likelihood)

    ipnp(xc, yc, xt)
