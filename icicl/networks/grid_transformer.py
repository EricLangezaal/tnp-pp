from typing import Optional

import einops
import warnings
import torch
from torch import nn
from check_shapes import check_shapes

from .attention_layers import MultiHeadCrossAttentionLayer
from .swin_attention import SWINAttentionLayer 
from .transformer import _get_clones
from .patch_encoders import PatchEncoder
from .transformer import TNPDTransformerEncoder

from ..utils.grids import nearest_gridded_neighbours, flatten_grid, unflatten_grid

class SWINTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        swin_layer: SWINAttentionLayer,
        patch_encoder: Optional[PatchEncoder] = None,
        top_k_ctot: Optional[int] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.swin_layers = _get_clones(swin_layer, num_layers)
        self.patch_encoder = patch_encoder
        self.top_k_ctot = top_k_ctot

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "mask: [m, nt, nc]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        zc: torch.Tensor,
        xt: torch.Tensor,
        zt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        if self.patch_encoder is not None:
            zc = self.patch_encoder(zc)
            xc = self.patch_encoder.average_input_locations(xc)

        for swin_layer, mhca_layer in zip(self.swin_layers, self.mhca_layers):
            zc = swin_layer(zc)

            if self.top_k_ctot is not None:
                num_batches, nt = zt.shape[:2]

                # (batch_size, n, k).
                nearest_idx = nearest_gridded_neighbours(xt, xc, k=self.top_k_ctot)

                zc, flat_to_grid_fn = flatten_grid(zc)
                batch_idx = (
                    torch.arange(num_batches)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, nt, self.top_k_ctot)
                )
                nearest_zc = zc[
                    batch_idx,
                    nearest_idx,
                ]
                zc = flat_to_grid_fn(zc)

                # Rearrange tokens.
                zt = einops.rearrange(zt, "b n e -> (b n) 1 e")
                nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")

                # Do the MHCA operation, reshape and return.
                zt = mhca_layer(zt, nearest_zc)

                zt = einops.rearrange(zt, "(b n) 1 e -> b n e", b=num_batches)
            else:
                # Flatten xc before cross-attending.
                grid_shape = zc.shape[1:-1]
                zc = flatten_grid(zc)
                zt = mhca_layer(zt, zc)
                zc = unflatten_grid(zc, grid_shape)

        return zt


class GridTransformerEncoder(TNPDTransformerEncoder):
    def __init__(
        self,
        *,
        patch_encoder: Optional[PatchEncoder] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_encoder = patch_encoder

    @check_shapes(
        "xc: [m, ..., d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.patch_encoder is not None:
            xc = self.patch_encoder(xc)
            
        xc = flatten_grid(xc)
        return super().forward(xc, xt, mask)