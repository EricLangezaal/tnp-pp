from typing import Optional, Tuple, Union

import einops
import warnings
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from check_shapes import check_shapes

from .attention_layers import MultiHeadCrossAttentionLayer, MultiHeadSelfAttentionLayer
from .swin_attention import SWINAttentionLayer 
from .transformer import _get_clones
from .patch_encoders import PatchEncoder
from .transformer import TNPDTransformerEncoder

from ..utils.grids import nearest_gridded_neighbours, flatten_grid, unflatten_grid, func_AvgPoolNd

class GridTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: Union[SWINAttentionLayer, MultiHeadSelfAttentionLayer],
        patch_encoder: Optional[PatchEncoder] = None,
        top_k_ctot: Optional[int] = None,
        roll_dims: Optional[Tuple[int,...]] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.patch_encoder = patch_encoder
        self.top_k_ctot = top_k_ctot
        self.roll_dims = roll_dims

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
            
            xc = func_AvgPoolNd(
                n=xc.shape[-1],
                input=xc.movedim(-1, 1),
                kernel_size=self.patch_encoder.conv.kernel_size,
                stride=self.patch_encoder.conv.stride,
            )
            xc = xc.movedim(1, -1)

        grid_shape = zc.shape[1:-1]
        zc = flatten_grid(zc)

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, SWINAttentionLayer):
                zc = unflatten_grid(zc, grid_shape)
                zc = mhsa_layer(zc)
                zc = flatten_grid(zc)
            else:
                zc = mhsa_layer(zc)

            if self.top_k_ctot is not None:
                num_batches, nt = zt.shape[:2]

                # (batch_size, n, k).
                nearest_idx, mask = nearest_gridded_neighbours(
                    xt, xc, k=self.top_k_ctot, roll_dims=self.roll_dims
                )

                batch_idx = (
                    torch.arange(num_batches)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, nt, nearest_idx.shape[-1])
                )
                nearest_zc = zc[
                    batch_idx,
                    nearest_idx,
                ]

                # Rearrange tokens.
                zt = einops.rearrange(zt, "b n e -> (b n) 1 e")
                nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")
                mask = einops.rearrange(mask, "b n e -> (b n) 1 e")

                # Do the MHCA operation, reshape and return.
                with sdpa_kernel(SDPBackend.MATH): 
                   zt = mhca_layer(zt, nearest_zc, mask=mask)

                zt = einops.rearrange(zt, "(b n) 1 e -> b n e", b=num_batches)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                   zt = mhca_layer(zt, zc)

        return zt
