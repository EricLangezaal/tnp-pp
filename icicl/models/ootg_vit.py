import torch
from torch import nn
from abc import ABC
from typing import Tuple

from ..utils.conv import make_grid, unflatten_grid, flatten_grid, convNd
from ..models.ootg_tnp import OOTG_TNPDEncoder, OOTGSetConvTNPDEncoder, OOTG_MHCA_TNPDEncoder

class OOTG_ViTEncoder(nn.Module, ABC):
    """
    Implements a very basic ViT encoding without positional embeddings

    This relies on applying convolutions to coarsen the grid, which only works for grids that span up to 3 dimensions
    The dimensionality of the data is unrestricted.
    """

    def __init__(
            self,
            *,
            grid_range: Tuple[Tuple[float, float], ...],
            points_per_unit: int,
            patch_size: int,
            embed_dim: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_shape = make_grid(grid_range[:, :1], grid_range[:, 1:2], points_per_unit, 0).shape[1:-1]
        self.patcher = convNd(n=len(self.grid_shape), in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)        

    def coarsen_grid(self, z: torch.Tensor) -> torch.Tensor:
        # z will be of shape (batch, num_on_grid, embed_dim)
        z = unflatten_grid(z, grid_shape=self.grid_shape)
        # move 'channels' (i.e embed_dim) right after batch
        z = z.movedim(-1, 1)
        z = self.patcher(z)
        # move 'channels' (i.e embed_dim) to end again
        z = z.movedim(1, -1)

        return flatten_grid(z)
    

class OOTGSetConvViTEncoder(OOTGSetConvTNPDEncoder, OOTG_ViTEncoder):
    def grid_encode(self, **kwargs) -> torch.Tensor:
        zc = super().grid_encode(**kwargs)
        zc = super().coarsen_grid(zc)
        return zc
    
    
class OOTG_MHCA_ViTEncoder(OOTG_MHCA_TNPDEncoder, OOTG_ViTEncoder):
    def grid_encode(self, **kwargs) -> torch.Tensor:
        zc = super().grid_encode(**kwargs)
        zc = super().coarsen_grid(zc)
        return zc
