import copy
from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn

from .teattention_layers import MultiHeadSelfTEAttentionLayer


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


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
