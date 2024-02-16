from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.batch import compress_batch_dimensions
from .attention_layers import MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer
from .transformer import _get_clones


class StochasticTNPDTransformerEncoder(nn.Module):
    def __init__(
        self,
        mhca_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        final_layer_cross_attention: bool = False,
    ):
        super().__init__()

        self.mhca_loc_layers = _get_clones(mhca_layer, num_layers)
        self.mhca_std_layers = _get_clones(mhca_layer, num_layers)
        self.final_layer_cross_attention = final_layer_cross_attention

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, s, nt, d]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        xc, xt, mask, uncompress_xt = _expand_dims(xc, xt, mask, num_samples)
        for mhca_loc_layer, mhca_std_layer in zip(
            self.mhca_loc_layers, self.mhca_std_layers
        ):
            if not self.final_layer_cross_attention:
                # Use the mean layer here.
                xt = mhca_loc_layer(xt, xc, mask)

            xc_loc = mhca_loc_layer(xc, xc)
            xc_raw_std = mhca_std_layer(xc, xc)
            xc_std = nn.functional.softplus(xc_raw_std)

            # Sample xc.
            xc = xc_loc + xc_std * torch.randn_like(xc_std)

        if self.final_layer_cross_attention:
            xt = self.mhca_layers[-1](xt, xc, mask)

        # Returns to shape (m, num_samples, nt, dx).
        xt = uncompress_xt(xt)

        return xt


class StochasticNestedPerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        min_latent_std: float = 0.0,
    ):
        super().__init__()

        assert mhsa_layer.embed_dim == mhca_ctoq_layer.embed_dim, "embed_dim mismatch."
        assert mhsa_layer.embed_dim == mhca_qtot_layer.embed_dim, "embed_dim mismatch."

        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_loc_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_ctoq_std_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        self.min_latent_std = min_latent_std

    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nq, n]",
        "return: [m, s, nq, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0] * num_samples)
        xc, xt, mask, uncompress_xt = _expand_dims(xc, xt, mask, num_samples)
        for (
            mhsa_layer,
            mhca_ctoq_loc_layer,
            mhca_ctoq_std_layer,
            mhca_qtot_layer,
        ) in zip(
            self.mhsa_layers,
            self.mhca_ctoq_loc_layers,
            self.mhca_ctoq_std_layers,
            self.mhca_qtot_layers,
        ):
            xq_loc = mhca_ctoq_loc_layer(xq, xc, mask)
            xq_raw_std = mhca_ctoq_std_layer(xq, xc, mask)
            xq_std = nn.functional.softplus(xq_raw_std) + self.min_latent_std

            # Sample xq.
            xq = xq_loc + xq_std * torch.randn_like(xq_std)

            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

        # Returns to shape (m, num_samples, nt, dx).
        xt = uncompress_xt(xt)

        return xt


class StochasticNestedISetTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
        min_latent_std: float = 0.0,
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

        self.mhca_ctoq_loc_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_ctoq_std_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

        self.min_latent_std = min_latent_std

    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nq, n]",
        "return: [m, s, nq, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        xt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0] * num_samples)
        xc, xt, mask, uncompress_xt = _expand_dims(xc, xt, mask, num_samples)
        for (
            mhca_ctoq_loc_layer,
            mhca_ctoq_std_layer,
            mhca_qtoc_layer,
            mhca_qtot_layer,
        ) in zip(
            self.mhca_ctoq_loc_layers,
            self.mhca_ctoq_std_layers,
            self.mhca_qtoc_layers,
            self.mhca_qtot_layers,
        ):
            xq_loc = mhca_ctoq_loc_layer(xq, xc, mask)
            xq_raw_std = mhca_ctoq_std_layer(xq, xc, mask)
            xq_std = nn.functional.softplus(xq_raw_std) + self.min_latent_std

            # Sample xq.
            xq = xq_loc + xq_std * torch.randn_like(xq_std)

            xc = mhca_qtoc_layer(xc, xq)
            xt = mhca_qtot_layer(xt, xq)

        # Returns to shape (m, num_samples, nt, dx).
        xt = uncompress_xt(xt)

        return xt


@check_shapes(
    "xc: [m, nc, dx]",
    "xt: [m, nt, dx]",
    "mask: [m, nq, n]",
)
def _expand_dims(
    xc: torch.Tensor,
    xt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_samples: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable]:
    xc = einops.repeat(xc, "m n d -> m s n d", s=num_samples)
    xt = einops.repeat(xt, "m n d -> m s n d", s=num_samples)

    if mask is not None:
        mask = einops.repeat(mask, "m n1 n2 -> m s n1 n2", s=num_samples)

    # Now compress.
    xc, _ = compress_batch_dimensions(xc, other_dims=2)
    xt, uncompress_xt = compress_batch_dimensions(xt, other_dims=2)

    if mask is not None:
        mask, _ = compress_batch_dimensions(mask, other_dims=2)

    return xc, xt, mask, uncompress_xt
