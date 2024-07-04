import math
from typing import Tuple

import torch
from check_shapes import check_shapes
from ..utils.grids import convNdModule
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: Tuple[int],
    ):
        super().__init__()

        self.conv = convNdModule(
            n=len(patch_size), 
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )   
    

    @check_shapes(
        "x: [m, ..., d]",
        "return: [m, ..., d]",
    )
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        grid_shape = torch.as_tensor(x.shape[1:-1])
        assert torch.all(
            grid_shape % torch.as_tensor(self.conv.kernel_size) == 0
        ), "Kernel size does not divide grid."

         # move 'channels' (i.e embed_dim) right after batch so shape is now (b, e, n1, n2, ..., ndim)
        x = x.movedim(-1, 1)
        x = self.conv(x)
        # move 'channels' (i.e embed_dim) to end again for shape  (b, n1/p1, n2/p2, ..., ndim/pdim, e)
        x = x.movedim(1, -1)

        return x