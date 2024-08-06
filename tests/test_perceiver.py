import torch

from tnp.networks.attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from tnp.networks.transformer import (
    NestedPerceiverEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
)


def test_perceiver_encoder():
    # Test parameters.
    m = 4
    n = 64
    num_latents = 16
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 32
    num_layers = 3

    x = torch.randn(m, n, embed_dim)

    mhsa_layer = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    mhca_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    perceiver_encoder = PerceiverEncoder(
        num_latents=num_latents,
        mhsa_layer=mhsa_layer,
        mhca_layer=mhca_layer,
        num_layers=num_layers,
    )

    perceiver_encoder(x)


def test_perceiver_decoder():
    # Test parameters.
    m = 4
    n = 64
    num_latents = 16
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 32
    num_layers = 3

    x = torch.randn(m, n, embed_dim)
    xq = torch.randn(m, num_latents, embed_dim)

    mhca_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    perceiver_decoder = PerceiverDecoder(
        mhca_layer=mhca_layer,
        num_layers=num_layers,
    )

    perceiver_decoder(x, xq)


def test_nested_perceiver_encoder():
    # Test parameters.
    m = 4
    nc = 64
    nt = 16
    num_latents = 16
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 32
    num_layers = 3

    xc = torch.randn(m, nc, embed_dim)
    xt = torch.randn(m, nt, embed_dim)

    mhsa_layer = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    mhca_ctoq_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    mhca_qtot_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )

    nested_perceiver_encoder = NestedPerceiverEncoder(
        num_latents=num_latents,
        mhsa_layer=mhsa_layer,
        mhca_ctoq_layer=mhca_ctoq_layer,
        mhca_qtot_layer=mhca_qtot_layer,
        num_layers=num_layers,
    )

    nested_perceiver_encoder(xc, xt)
