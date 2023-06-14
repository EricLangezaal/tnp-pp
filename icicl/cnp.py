from typing import Optional, Tuple

import torch
from torch import nn

from .deepset import DeepSet
from .models import NP
from .nn import MLP


class CNPDecoder(nn.Module):
    """Represents the decoder for a conditional neural process."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(
        self, z_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.distributions.Distribution:
        # Use same context for each prediction.
        z_c = z_c.unsqueeze(0).repeat(x_t.shape[0], 1)
        return self.mlp(torch.cat((z_c, x_t), dim=-1))


def construct_cnp(
    x_dim: int,
    y_dim: int,
    r_dim: int,
    encoder_layers: Optional[Tuple[int, ...]] = None,
    encoder_num_layers: Optional[Tuple[int, ...]] = None,
    encoder_width: Optional[int] = None,
    encoder_nonlinearity=None,
    decoder_layers: Optional[Tuple[int, ...]] = None,
    decoder_num_layers: Optional[Tuple[int, ...]] = None,
    decoder_width: Optional[int] = None,
    decoder_nonlinearity=None,
    likelihood=None,
):
    encoder_mlp = MLP(
        in_dim=x_dim + y_dim,
        out_dim=r_dim,
        layers=encoder_layers,
        num_layers=encoder_num_layers,
        width=encoder_width,
        nonlinearity=encoder_nonlinearity,
    )
    encoder = DeepSet(phi=encoder_mlp)

    decoder_mlp = MLP(
        in_dim=r_dim + x_dim,
        out_dim=y_dim,
        layers=decoder_layers,
        num_layers=decoder_num_layers,
        width=decoder_width,
        nonlinearity=decoder_nonlinearity,
    )

    decoder = CNPDecoder(
        mlp=decoder_mlp,
    )

    return NP(encoder, decoder, likelihood)
