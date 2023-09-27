import torch

from icicl.networks.attention import MultiHeadSelfAttention


def test_mhsa():
    # Test parameters.
    m = 16
    n = 64
    embed_dim = 32
    num_heads = 8
    head_dim = 3

    x = torch.randn(m, n, embed_dim)

    mhsa = MultiHeadSelfAttention(
        embed_dim=embed_dim, num_heads=num_heads, head_dim=head_dim
    )
    mhsa(x)
