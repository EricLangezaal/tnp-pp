import torch
from torch import nn
from check_shapes import check_shapes

from ..utils.conv import unflatten_grid, flatten_grid, convNd
from ..utils.helpers import preprocess_observations
from ..models.ootg_tnp import OOTG_TNPDEncoder, OOTGSetConvTNPDEncoder, OOTG_MHCA_TNPDEncoder


class OOTG_ViTEncoder(OOTG_TNPDEncoder):
    """
    Implements a very basic ViT encoding without positional embeddings

    This relies on applying convolutions to coarsen the grid, which only works for grids that span up to 3 dimensions
    The dimensionality of the data is unrestricted.
    """

    def __init__(
            self,
            *,
            patch_size: int,
            dim: int,
            embed_dim: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.patcher = convNd(n=dim, in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)        

    def coarsen_grid(self, z: torch.Tensor) -> torch.Tensor:
        # z will be of shape (batch, num_on_grid, embed_dim)
        z = unflatten_grid(z, dim=self.dim)
        # move 'channels' (i.e embed_dim) right after batch
        z = z.movedim(-1, 1)
        z = self.patcher(z)
        # move 'channels' (i.e embed_dim) to end again
        z = z.movedim(1, -1)

        return flatten_grid(z)
    

class OOTGSetConvViTEncoder(OOTGSetConvTNPDEncoder, OOTG_ViTEncoder):
    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        ignore_on_grid: bool = False,
    ) -> torch.Tensor:
        # this will make yc's last dimension 2.
        xc, yc = self.grid_encode(xc_off_grid=xc_off_grid, yc_off_grid=yc_off_grid, xc_on_grid=xc_on_grid, yc_on_grid=yc_on_grid)
        if ignore_on_grid:
            yc = yc[..., 1:]
        # this makes yc's last dimension 3.
        yc, yt = preprocess_observations(xt, yc) # 
        zc = torch.cat((xc, yc), dim=-1)
        # So zc is 3 + xdim
        zc = self.xy_encoder(zc)
        zc = self.coarsen_grid(zc)   

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        return self.transformer_encoder(zc, zt)
    
    
class OOTG_MHCA_ViTEncoder(OOTG_MHCA_TNPDEncoder, OOTG_ViTEncoder):
    def grid_encode(self, **kwargs) -> torch.Tensor:
        zc = OOTG_MHCA_TNPDEncoder.grid_encode(**kwargs)
        zc = OOTG_ViTEncoder.coarsen_grid(zc)
        return zc
