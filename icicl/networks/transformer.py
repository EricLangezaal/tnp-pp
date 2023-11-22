import copy
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
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for mhca_layer in self.mhca_layers:
            xt = mhca_layer(xt, xc, mask)
            xc = mhca_layer(xc, xc)

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


class ICNestedPerceiverEncoder(BaseNestedPerceiverEncoder):
    def __init__(
        self,
        num_latents: int,
        num_ic_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        super().__init__(
            num_latents, mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer, num_layers
        )

        embed_dim = mhsa_layer.embed_dim
        self.ic_latents = nn.Parameter(torch.randn(num_ic_latents, embed_dim))

        self.ic_mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, dx]",
        "xic: [m, nic, ncic, dx]",
        "xt: [m, nt, dx]",
        "return: [m, nq, d]",
    )
    def forward(
        self, xc: torch.Tensor, xic: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        xqic = einops.repeat(
            self.ic_latents, "l e -> m n l e", m=xic.shape[0], n=xic.shape[1]
        )
        # shape (m x nic, ncic, dx).
        xic, _ = compress_batch_dimensions(xic, other_dims=2)
        for (mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer, ic_mhca_qtot_layer,) in zip(
            self.mhsa_layers,
            self.mhca_ctoq_layers,
            self.mhca_qtot_layers,
            self.ic_mhca_qtot_layers,
        ):
            # Cross-attention between latents and context set.
            xq = mhca_ctoq_layer(xq, xc)
            # Self-attention between latents.
            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

            # Attention with in-context datasets.
            xqic, xqic_uncompress = compress_batch_dimensions(xqic, other_dims=2)
            xqic = mhca_ctoq_layer(xqic, xic)
            xqic = mhsa_layer(xqic)
            xqic = xqic_uncompress(xqic)

            # shape (m, nic x ncic, dx)
            xqic = einops.rearrange(xqic, "m n l e -> m (n l) e")
            xt = ic_mhca_qtot_layer(xt, xqic)
            xqic = einops.rearrange(
                xqic, "m (n l) e -> m n l e", l=self.ic_latents.shape[0]
            )

        return xt


# class ICNestedPerceiverEncoder(BaseNestedPerceiverEncoder):
#     def __init__(
#         self,
#         num_latents: int,
#         num_ic_latents: int,
#         mhsa_layer: MultiHeadSelfAttentionLayer,
#         mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
#         mhca_qtot_layer: MultiHeadCrossAttentionLayer,
#         ic_mhca_qtoq_layer: MultiHeadCrossAttentionLayer,
#         num_layers: int,
#     ):
#         super().__init__(
#             num_latents, mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer, num_layers
#         )

#         embed_dim = mhsa_layer.embed_dim
#         self.ic_latents = nn.Parameter(torch.randn(num_ic_latents, embed_dim))

#         self.ic_mhca_qtoq_layers = _get_clones(ic_mhca_qtoq_layer, num_layers)

#     @check_shapes(
#         "xc: [m, nc, dx]",
#         "xic: [m, nic, ncic, dx]",
#         "xt: [m, nt, dx]",
#         "return: [m, nq, d]",
#     )
#     def forward(
#         self, xc: torch.Tensor, xic: torch.Tensor, xt: torch.Tensor
#     ) -> torch.Tensor:
#         xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
#         xqic = einops.repeat(
#             self.ic_latents, "l e -> m n l e", m=xic.shape[0], n=xic.shape[1]
#         )
#         # shape (m x nic, ncic, dx).
#         xic, _ = compress_batch_dimensions(xic, other_dims=2)
#         for (
#             mhsa_layer,
#             mhca_ctoq_layer,
#             mhca_qtot_layer,
#             ic_mhca_qtoq_layer,
#         ) in zip(
#             self.mhsa_layers,
#             self.mhca_ctoq_layers,
#             self.mhca_qtot_layers,
#             self.ic_mhca_qtoq_layers,
#         ):
#             # Cross-attention between latents and context set.
#             xq = mhca_ctoq_layer(xq, xc)
#             # Self-attention between latents.
#             xq = mhsa_layer(xq)

#             # Cross-attention between in-context latents and context sets.
#             xqic, xqic_uncompress = compress_batch_dimensions(xqic, other_dims=2)
#             xqic = mhca_ctoq_layer(xqic, xic)
#             # Self-attention between in-context latents.
#             xqic = mhsa_layer(xqic)
#             xqic = xqic_uncompress(xqic)

#             # Cross-attention between latents and in-context latents.
#             xqic = einops.rearrange(xqic, "m n l e -> m (n l) e")
#             xq = ic_mhca_qtoq_layer(xq, xqic)
#             xqic = einops.rearrange(
#                 xqic, "m (n l) e -> m n l e", l=self.ic_latents.shape[0]
#             )

#             # Cross-attention between target set and latents.
#             xt = mhca_qtot_layer(xt, xq)

#         return xt


class ParallelNestedPerceiverEncoder(BaseNestedPerceiverEncoder):
    def __init__(self, *, mhca_layer: MultiHeadSelfAttentionLayer, **kwargs):
        super().__init__(**kwargs)

        self.mhca_layers = _get_clones(mhca_layer, len(self.mhsa_layers))

    @check_shapes(
        "xc: [m, nc, dx]",
        "xt: [m, nt, dx]",
        "mask: [m, nq, n]",
        "return: [m, nq, d]",
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        m = xc.shape[0]
        l = self.latents.shape[0]

        xq = einops.repeat(self.latents, "l e -> m l e", m=m)
        for mhsa_layer, mhca_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers,
            self.mhca_layers,
            self.mhca_ctoq_layers,
            self.mhca_qtot_layers,
        ):
            # Cross-attention between context and latent sets.
            xq = mhca_ctoq_layer(xq, xc, mask)
            xq = mhsa_layer(xq)

            # shape (1, m x nc, dx).
            xq = einops.rearrange(xq, "m l e -> 1 (m l) e")
            mhca_mask = torch.block_diag(*[torch.ones(l, l)] * m) > 0.5
            mhca_mask = einops.repeat(mhca_mask, "a b -> m a b", m=xq.shape[0])

            # Cross-attention betweeen latent sets.
            xq = mhca_layer(xq, mhca_mask)

            # Cross-attention between latent and target sets.
            xq = einops.rearrange(xq, "1 (m l) e -> m l e", m=m)
            xt = mhca_qtot_layer(xt, xq)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
