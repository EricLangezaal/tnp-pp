import pytest
import torch

from icicl.likelihoods.gaussian import NormalLikelihood
from icicl.models.tnp import TNPD, EfficientTNPDEncoder, TNPDDecoder, gen_tnpd_mask
from icicl.networks.attention_layers import MultiHeadCrossAttentionLayer
from icicl.networks.mlp import MLP
from icicl.networks.transformer import TNPDTransformerEncoder


@pytest.mark.parametrize("ndim", [1, 2])
def test_tnpd(ndim: int):
    # Test parameters.
    m = 16
    nc = 128
    nt = 64
    dy = 1

    # Encoder parameters.
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 128
    num_layers = 5

    xc = torch.randn(m, nc, ndim)
    yc = torch.randn(m, nc, 1)
    xt = torch.randn(m, nt, ndim)

    mhca_layer = MultiHeadCrossAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    transformer_encoder = TNPDTransformerEncoder(
        mhca_layer=mhca_layer, num_layers=num_layers
    )
    xy_encoder = MLP(dy + ndim + 1, embed_dim, num_layers=2, width=64)
    z_decoder = MLP(embed_dim, dy, num_layers=2, width=64)

    tnpd_encoder = EfficientTNPDEncoder(
        transformer_encoder=transformer_encoder, xy_encoder=xy_encoder
    )
    tnpd_decoder = TNPDDecoder(z_decoder=z_decoder)
    likelihood = NormalLikelihood(noise=1e-1)

    tnpd = TNPD(encoder=tnpd_encoder, decoder=tnpd_decoder, likelihood=likelihood)

    tnpd(xc, yc, xt)


def test_gen_tnpd_mask():
    nc = 2
    nt = 3
    m = 2
    ndim = 1

    xc = torch.randn(m, nc, ndim)
    xt = torch.randn(m, nt, ndim)
    mask = gen_tnpd_mask(xc, xt)

    for mask_ in mask:
        for i in range(nc):
            for j in range(nc):
                assert mask_[i, j] == False

            for j in range(nt):
                assert mask_[i, nc + j] == True

        for i in range(nt):
            for j in range(nc):
                assert mask_[nc + i, j] == False

            for j in range(nt):
                assert mask_[nc + i, nc + j] == True
