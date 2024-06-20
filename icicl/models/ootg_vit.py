import torch
from torch import nn
from check_shapes import check_shapes

from ..networks.grid_encoders import IdentityGridEncoder
from ..utils.conv import convNd
from ..utils.grids import flatten_grid
from ..models.ootg_tnp import OOTG_TNPDEncoder


class OOTG_ViTEncoder(OOTG_TNPDEncoder):
    """
    Implements a very basic ViT encoding without positional embeddings

    This relies on applying convolutions to coarsen the grid, which only works for grids that span up to 3 dimensions
    The dimensionality of the data is unrestricted.
    """

    def __init__(
            self,
            *,
            dim: int,
            patch_size: int,
            embed_dim: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert not isinstance(self.grid_encoder, IdentityGridEncoder)
        self.dim = dim
        self.patcher = convNd(n=dim, in_channels=embed_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)   

    @check_shapes("z: [b, ..., e]", "return: [b, n, e]")
    def prepare_context_tokens(
            self, 
            z: torch.Tensor
    ) -> torch.Tensor:
        assert z.ndim - 2 == self.dim, f"Embedded context should match grid shape, expected {self.dim} grid, got {z.ndim - 2} grid."
        # move 'channels' (i.e embed_dim) right after batch so shape is now (b, e, n1, n2, ..., ndim)
        z = z.movedim(-1, 1)
        z = self.patcher(z)
        # move 'channels' (i.e embed_dim) to end again for shape  (b, n1, n2, ..., ndim, e)
        z = z.movedim(1, -1)

        return flatten_grid(z) 
