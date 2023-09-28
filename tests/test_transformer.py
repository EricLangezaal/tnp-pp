import torch

from icicl.networks.transformer import MultiHeadSelfAttentionLayer, TransformerEncoder


def test_transformer():
    # Test parameters.
    m = 16
    n = 64
    embed_dim = 32
    num_heads = 8
    head_dim = 3
    feedforward_dim = 128
    num_layers = 5

    x = torch.randn(m, n, embed_dim)
    mask = torch.rand(m, n, n) > 0.5

    layer = MultiHeadSelfAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        feedforward_dim=feedforward_dim,
    )
    transformer = TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

    transformer(x, mask)
