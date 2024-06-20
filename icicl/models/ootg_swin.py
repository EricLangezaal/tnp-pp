from typing import Optional
import warnings

from check_shapes import check_shapes
import einops
from torch import nn
import torch

from ..models.ootg_tnp import OOTG_TNPDEncoder
from ..networks.grid_encoders import IdentityGridEncoder
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..networks.swin_attention import SWINAttentionLayer
from ..networks.transformer import _get_clones
from ..utils.grids import unflatten_grid, flatten_grid


class SWINTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        swin_layer: SWINAttentionLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.swin_layers = _get_clones(swin_layer, num_layers)

    @check_shapes(
        "xc: [m, ..., d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        grid_shape = xc.shape[1:-1]      

        for swin_layer, mhca_layer in zip(self.swin_layers, self.mhca_layers):
            xc = swin_layer(xc)

            # Flatten xc before cross-attending.
            xc = flatten_grid(xc)
            xt = mhca_layer(xt, xc)
            xc = unflatten_grid(xc, grid_shape)

        return xt
    

class OOTG_SwinEncoder(OOTG_TNPDEncoder):

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(self.transformer_encoder, SWINTransformerEncoder)
        assert not isinstance(self.grid_encoder, IdentityGridEncoder)

    @check_shapes("z: [b, ..., e]", "return: [b, ..., e]")
    def prepare_context_tokens(
            self, 
            z: torch.Tensor
    ) -> torch.Tensor:
        # do not flatten here, as SWIN wants a grid
        return z