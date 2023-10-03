import torch

from icicl.networks.attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from icicl.networks.transformer import SPINEncoder


def test_spin_encoder():
    # Test parameters.
    m = 4
    n = 64
    num_latent_features = 3
    num_latent_datapoints = 16
    ndim = 5
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 32
    num_layers = 3

    x = torch.randn(m, n, ndim, embed_dim)

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
        embed_dim=num_latent_features * embed_dim,
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

    spin_encoder(x)
