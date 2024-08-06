from typing import Tuple

import torch
from torch import nn

from ..utils.spherical_harmonics.spherical_harmonics_ylm import spherical_harmonics


class SineActivation(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()

        # TODO: need to change for first layer?
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SphericalHarmonicsEmbedding(nn.Module):
    def __init__(
        self, num_legendre_polys: int = 10, lonlat_dims: Tuple[int] = (-1, -2)
    ):
        super().__init__()

        self.num_legendre_polys = int(num_legendre_polys)
        self.embed_dim = self.num_legendre_polys**2
        self.lonlat_dims = lonlat_dims

        self.spherical_harmonics = spherical_harmonics

    def forward(self, x: torch.Tensor):
        """
        Assumes x[..., lonlat_dims] = (lon, lat) where lon is in [-180, 180] and lat is in
        [-90, 90].
        """

        if x.shape[-1] > 2:
            x_other = torch.stack(
                [
                    x[..., dim]
                    for dim in range(x.shape[-1])
                    if dim not in self.lonlat_dims
                ]
            )
        else:
            x_other = None

        lon, lat = x[..., self.lonlat_dims[0]], x[..., self.lonlat_dims[1]]

        # Assumes phi is in [-pi, pi] and lat is in [-pi / 2, pi / 2].
        phi, theta = torch.deg2rad(lon), torch.deg2rad(lat)

        # Compute the spherical harmonics.
        sh_list = []
        for l in range(self.num_legendre_polys):
            for m in range(-l, l + 1):
                sh = self.spherical_harmonics(m, l, phi, theta)
                if isinstance(sh, float):
                    sh = sh * torch.ones_like(phi)
                sh_list.append(sh)

        out = torch.stack(sh_list, dim=-1)

        if x_other is not None:
            out = torch.cat((x_other, out), dim=-1)

        return out