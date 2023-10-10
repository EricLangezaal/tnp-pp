import copy
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .teattention_layers import (
    MultiHeadCrossTEAttentionLayer,
    MultiHeadSelfTEAttentionLayer,
)


class TETransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: MultiHeadSelfTEAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]", "mask: [m, n, n]", "return: [m, n, dy]"
    )
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            y = layer(x, y, mask)

        return y


class TEPerceiverEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_latents: int,
        mhsa_layer: MultiHeadSelfTEAttentionLayer,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert mhsa_layer.embed_dim == mhca_layer.embed_dim, "embed_dim mismatch."

        embed_dim = mhsa_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes(
        "x: [m, n, d]", "y: [m, n, dy]", "mask: [m, nq, n]", "return: [m, nq, dy]"
    )
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        yq = einops.repeat(self.latent_tokens, "l e -> m l e", m=x.shape[0])
        xq = einops.repeat(self.latent_inputs, "l d -> m l d", m=x.shape[0])
        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            yq = mhca_layer(xq, x, yq, y, mask)
            yq = mhsa_layer(xq, yq)

        return yq


class TEPerceiverDecoder(nn.Module):
    def __init__(
        self,
        mhca_layer: MultiHeadCrossTEAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes(
        "x: [m, n, d]",
        "xq: [m, nq, d]",
        "yq: [m, nq, dy]",
        "mask: [m, n, nq]",
        "return: [m, n, d]",
    )
    def forward(
        self,
        x: torch.Tensor,
        xq: torch.Tensor,
        y: torch.Tensor,
        yq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for mhca_layer in self.mhca_layers:
            y = mhca_layer(x, xq, y, yq, mask)

        return y


class BaseNestedTEPerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        num_latents: int,
        mhsa_layer: MultiHeadSelfTEAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossTEAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossTEAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert mhsa_layer.embed_dim == mhca_ctoq_layer.embed_dim, "embed_dim mismatch."
        assert mhsa_layer.embed_dim == mhca_qtot_layer.embed_dim, "embed_dim mismatch."

        embed_dim = mhsa_layer.embed_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, embed_dim))
        self.latent_inputs = nn.Parameter(torch.randn(num_latents, dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class NestedTEPerceiverEncoder(BaseNestedTEPerceiverEncoder):
    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "yc: [m, nc, dy]",
        "yt: [m, nt, dy]",
        "mask: [m, nq, n]",
        "return: [m, nq, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        yc: torch.Tensor,
        yt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        yq = einops.repeat(self.latent_tokens, "l e -> m l e", m=xc.shape[0])
        xq = einops.repeat(self.latent_inputs, "l d -> m l d", m=xc.shape[0])
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            yq = mhca_ctoq_layer(xq, xc, yq, yc, mask)
            yq = mhsa_layer(xq, yq)
            yt = mhca_qtot_layer(xt, xq, yt, yq)

        return yt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
