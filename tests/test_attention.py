import torch

from icicl.networks.attention import MultiHeadCrossAttention, MultiHeadSelfAttention


def test_mhsa():
    # Test parameters.
    m = 16
    n = 64
    embed_dim = 32
    num_heads = 8
    head_dim = 3

    x = torch.randn(m, n, embed_dim)
    mask = torch.ones(m, n, n) < 0.5

    mhsa = MultiHeadSelfAttention(
        embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim
    )
    mhsa(x, mask=mask)


def test_mhca():
    # Test parameters.
    m = 16
    nkv = 64
    nq = 8
    embed_dim = 32
    num_heads = 8
    head_dim = 3

    xkv = torch.randn(m, nkv, embed_dim)
    xq = torch.randn(m, nq, embed_dim)
    mask = torch.ones(m, nq, nkv) < 0.5

    mhca = MultiHeadCrossAttention(
        embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim
    )
    mhca(xq, xkv, mask=mask)
