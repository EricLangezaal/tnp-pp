import copy
import random
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.batch import compress_batch_dimensions, compress_data_dimensions
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
        mhca_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        final_layer_cross_attention: bool = False,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.final_layer_cross_attention = final_layer_cross_attention

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhca_layer in self.mhca_layers:
            if not self.final_layer_cross_attention:
                xt = mhca_layer(xt, xc)

            xc = mhca_layer(xc, xc)

        if self.final_layer_cross_attention:
            xt = self.mhca_layers[-1](xt, xc)

        return xt


class TNPDTransformerEncoderNotShared(nn.Module):
    def __init__(
        self,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        final_layer_cross_attention: bool = False,
    ):
        super().__init__()

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.final_layer_cross_attention = final_layer_cross_attention

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nc, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            xc = mhsa_layer(xc)

            if not self.final_layer_cross_attention:
                xt = mhca_layer(xt, xc)

        if self.final_layer_cross_attention:
            xt = self.mhca_layers[-1](xt, xc)

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


class RandomLatentsNestedISetTransformerEncoder(nn.Module):
    def __init__(
        self,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        min_num_latents: int = 1,
        max_num_latents: int = 32,
        random_projection: bool = False,
    ):
        super().__init__()

        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtoc_layer.embed_dim
        ), "embed_dim mismatch."
        assert (
            mhca_ctoq_layer.embed_dim == mhca_qtot_layer.embed_dim
        ), "embed_dim mismatch."

        self.embed_dim = mhca_ctoq_layer.embed_dim
        self.min_num_latents = min_num_latents
        self.max_num_latents = max_num_latents
        self.random_projection = random_projection

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Initialise with randomly sampled latents.
        num_latents = random.choice(
            list(range(self.min_num_latents, self.max_num_latents + 1))
        )
        if self.random_projection:
            rand_mat = torch.randn((xc.shape[0], num_latents, xc.shape[1]))
            xq = rand_mat @ xc
        else:
            xq = torch.randn((xc.shape[0], num_latents, self.embed_dim))

        for mhca_ctoq_layer, mhca_qtoc_layer, mhca_qtot_layer in zip(
            self.mhca_ctoq_layers, self.mhca_qtoc_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc, mask)
            xc = mhca_qtoc_layer(xc, xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


class SPINEncoder(nn.Module):
    def __init__(
        self,
        num_latent_features: int,
        num_latent_datapoints: int,
        xaba_layer: MultiHeadCrossAttentionLayer,
        abla_layer: MultiHeadSelfAttentionLayer,
        xabd_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert xaba_layer.embed_dim == abla_layer.embed_dim, "embed_dim mismatch."
        assert (
            xabd_layer.embed_dim == abla_layer.embed_dim * num_latent_features
        ), "embed_dim mismatch."

        embed_dim = xaba_layer.embed_dim
        self.latent_features = nn.Parameter(torch.randn(num_latent_features, embed_dim))
        self.latent_datapoints = nn.Parameter(
            torch.randn(num_latent_datapoints, embed_dim * num_latent_features)
        )

        self.xaba_layers = _get_clones(xaba_layer, num_layers)
        self.abla_layers = _get_clones(abla_layer, num_layers)
        self.xabd_layers = _get_clones(xabd_layer, num_layers)

    @check_shapes("x: [m, n, dx, e]", "mask: [m, nq, n]", "return: [m, nq, fe]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        xqf = einops.repeat(
            self.latent_features, "f e -> m n f e", m=x.shape[0], n=x.shape[1]
        )
        xqd = einops.repeat(self.latent_datapoints, "nq fe -> m nq fe", m=x.shape[0])

        # shape (m x n, dx, e).
        x, _ = compress_batch_dimensions(x, other_dims=2)

        for xaba_layer, abla_layer, xabd_layer in zip(
            self.xaba_layers, self.abla_layers, self.xabd_layers
        ):
            # shape (m x n, nqf, e).
            xqf, xqf_uncompress = compress_batch_dimensions(xqf, other_dims=2)

            xqf = xaba_layer(xqf, x)
            xqf = abla_layer(xqf)
            xqf = xqf_uncompress(xqf)

            # shape (m, n, nqf x e).
            xqf, xqf_uncompress = compress_data_dimensions(xqf, other_dims=2)

            # shape (m, nqd, nqf x e).
            xqd = xabd_layer(xqd, xqf, mask=mask)

            xqf = xqf_uncompress(xqf)

        return xqd


class SPINDecoder(nn.Module):
    def __init__(
        self,
        num_latent_features: int,
        xaba_layer: MultiHeadCrossAttentionLayer,
        abla_layer: MultiHeadSelfAttentionLayer,
        xabd_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        assert xaba_layer.embed_dim == abla_layer.embed_dim, "embed_dim mismatch."
        assert (
            xabd_layer.embed_dim == xaba_layer.embed_dim * num_latent_features
        ), "embed_dim mismatch."

        embed_dim = xaba_layer.embed_dim
        self.latent_features = nn.Parameter(torch.randn(num_latent_features, embed_dim))

        self.xaba_layers = _get_clones(xaba_layer, num_layers)
        self.abla_layers = _get_clones(abla_layer, num_layers)
        self.xabd_layers = _get_clones(xabd_layer, num_layers)

    @check_shapes(
        "xt: [m, n, d, e]", "xqc: [m, nq, dz]", "mask: [m, n, nq]", "return: [m, n, dz]"
    )
    def forward(
        self, xt: torch.Tensor, xqc: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        _ = mask

        xqt = einops.repeat(
            self.latent_features, "f e -> m n f e", m=xt.shape[0], n=xt.shape[1]
        )

        # shape (m, n, f, e).
        xqt, xqt_uncompress = compress_batch_dimensions(xqt, other_dims=2)
        xt, _ = compress_batch_dimensions(xt, other_dims=2)

        # Obtain latent attribute embeddings.
        for (
            xaba_layer,
            abla_layer,
        ) in zip(self.xaba_layers, self.abla_layers):
            xqt = xaba_layer(xqt, xt)
            xqt = abla_layer(xqt)

        xqt = xqt_uncompress(xqt)
        xqt, _ = compress_data_dimensions(xqt, other_dims=2)
        # Cross-attention with context representation.
        for xabd_layer in self.xabd_layers:
            xqt = xabd_layer(xqt, xqc)

        # shape (m, n, dz).
        return xqt


class ARTNPDTransformerEncoder(TNPDTransformerEncoder):
    xc_cache: dict[str, torch.Tensor] = {}

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nc, nc]", "return: [m, nt, d]"
    )
    def ar_forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Doesn't update tokens stored in xc_cache, and only uses xc_cache to update xc tokens."""
        if not bool(self.xc_cache):
            # Cache is empty: just do usual thing, storing xc in cache at each layer.
            for i, mhca_layer in enumerate(self.mhca_layers):
                self.xc_cache[f"layer-{i}"] = xc.clone().detach()
                if not self.final_layer_cross_attention:
                    xt = mhca_layer(xt, xc)

                # Mask can be used to apply any masking in first round.
                xc = mhca_layer(xc, xc, mask=self.mask)

            self.xc_cache[f"layer-{len(self.mhca_layers) - 1}"] = xc.clone().detach()

            if self.final_layer_cross_attention:
                xt = self.mhca_layers[-1](xt, xc)

        else:
            # User xc_cache to update xc tokens, then use both to update xt.
            # Also update xc_cache...
            xc_cache_update: dict[str, torch.Tensor] = {}
            for i, mhca_layer in enumerate(self.mhca_layers):
                xc_cache_update[f"layer-{i}"] = xc.clone().detach()
                # Set of tokens used to update xt.
                xc_ = torch.cat((xc, self.xc_cache[f"layer-{i}"]), dim=-2)
                if not self.final_layer_cross_attention:
                    xt = mhca_layer(xt, xc_)

                # Update xc using xc_cache.
                xc = mhca_layer(xc, self.xc_cache[f"layer-{i}"], mask)

            xc_cache_update[f"layer-{len(self.mhca_layers) - 1}"] = xc.clone().detach()

            if self.final_layer_cross_attention:
                xc_ = torch.cat(
                    (xc, self.xc_cache[f"layer-{len(self.mhca_layers) - 1}"]), dim=-2
                )
                xt = self.mhca_layers[-1](xt, xc_)

            # Update cache with xc_cache_update.
            for layer in self.xc_cache:
                self.xc_cache[layer] = torch.cat(
                    (self.xc_cache[layer], xc_cache_update[layer]), dim=-2
                )

        return xt

    def clear_cache(self):
        self.xc_cache = {}


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
