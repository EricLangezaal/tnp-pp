import copy
import random
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for mhsa_layer in self.mhsa_layers:
            x = mhsa_layer(x, mask)

        return x


class TNPDTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: Optional[MultiHeadSelfAttentionLayer] = None,
        final_layer_cross_attention: bool = False,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )
        self.final_layer_cross_attention = final_layer_cross_attention

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for i, (mhsa_layer, mhca_layer) in enumerate(
            zip(self.mhsa_layers, self.mhca_layers)
        ):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc)
            elif isinstance(mhsa_layer, MultiHeadCrossAttentionLayer):
                xc = mhsa_layer(xc, xc)

            if (not self.final_layer_cross_attention) or (
                i == len(self.mhsa_layers) - 1
            ):
                xt = mhca_layer(xt, xc)

        return xt


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert mhsa_layer.embed_dim == mhca_layer.embed_dim, "embed_dim mismatch."

        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes("x: [m, n, d]", "mask: [m, nq, n]", "return: [m, nq, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xq = self.latents.unsqueeze(0).repeat(x.shape[0], 1, 1)
        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            xq = mhca_layer(xq, x, mask)
            xq = mhsa_layer(xq)

        return xq


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        mhca_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes(
        "x: [m, n, d]", "xq: [m, nq, d]", "mask: [m, n, nq]", "return: [m, n, d]"
    )
    def forward(
        self, x: torch.Tensor, xq: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for mhca_layer in self.mhca_layers:
            x = mhca_layer(x, xq, mask)

        return x


class ISetTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtoc_layer.embed_dim
        ), "embed_dim mismatch."

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.num_layers = num_layers

    @check_shapes("x: [m, n, d]", "mask: [m, nq, n]", "return: [m, nq, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xq = self.latents.unsqueeze(0).repeat(x.shape[0], 1, 1)
        for mhca_ctoq_layer, mhca_qtoc_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers
        ):
            xq = mhca_ctoq_layer(xq, x, mask)
            x = mhca_qtoc_layer(x, xq)

        return x


class BaseNestedPerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert mhsa_layer.embed_dim == mhca_ctoq_layer.embed_dim, "embed_dim mismatch."
        assert mhsa_layer.embed_dim == mhca_qtot_layer.embed_dim, "embed_dim mismatch."

        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class NestedPerceiverEncoder(BaseNestedPerceiverEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc, mask)
            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


class NestedISetTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtoc_layer.embed_dim
        ), "embed_dim mismatch."
        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtot_layer.embed_dim
        ), "embed_dim mismatch."

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for mhca_ctoq_layer, mhca_qtoc_layer, mhca_qtot_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc, mask)
            xc = mhca_qtoc_layer(xc, xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
