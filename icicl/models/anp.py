import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.attention_layers import (
    MultiHeadAttentionLayer,
    MultiHeadCrossAttentionLayer,
)
from ..networks.transformer import TransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess
from .tnp import TNPDDecoder


class ANPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        mha_layer: MultiHeadAttentionLayer,
        xy_encoder: TransformerEncoder,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.mha_layer = mha_layer
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        # Self-attention layers on context tokens.
        zc = self.transformer_encoder(zc)

        # Cross-attention layer with input locations as query/keys.
        zt = self.mha_layer(xt, xc, zc)

        return zt


class SensibleANPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        mhca_layer: MultiHeadCrossAttentionLayer,
        xy_encoder: TransformerEncoder,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.mhca_layer = mhca_layer
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        # Self-attention layers on context tokens.
        zc = self.transformer_encoder(zc)

        # Cross-attention layer with input locations as query/keys.
        zt = self.mhca_layer(zt, zc)

        return zt


class ANP(ConditionalNeuralProcess):
    def __init__(
        self, encoder: ANPEncoder, decoder: TNPDDecoder, likelihood: nn.Module
    ):
        super().__init__(encoder, decoder, likelihood)


class SensibleANP(ConditionalNeuralProcess):
    def __init__(
        self, encoder: SensibleANPEncoder, decoder: TNPDDecoder, likelihood: nn.Module
    ):
        super().__init__(encoder, decoder, likelihood)
