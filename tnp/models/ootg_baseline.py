from scipy.interpolate import LinearNDInterpolator, interpn
import numpy as np
import torch
from torch import nn
import warnings

from ..networks.mlp import MLP
from ..data.on_off_grid import DataModality

class InterpBaselineEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.mix_layer = MLP(2, 1, layers=tuple())

    def forward(
        self, 
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        if used_modality != DataModality.BOTH:
            warnings.warn("Baseline will always use both modalities")
        
        device = xc_off_grid.device

        xc_off_grid = xc_off_grid.numpy(force=True) 
        yc_off_grid = yc_off_grid.numpy(force=True)
        xc_on_grid = xc_on_grid.numpy(force=True)
        yc_on_grid = yc_on_grid.numpy(force=True)
        xt = xt.numpy(force=True)

        yt_off = []
        yt_on = []
        for bxc_off, byc_off, bxc_on, byc_on, bxt in zip(xc_off_grid, yc_off_grid, xc_on_grid, yc_on_grid, xt):
            off_interp = LinearNDInterpolator(bxc_off, byc_off, fill_value=byc_off.mean())
            yt_off.append(off_interp(bxt))

            on_interp = interpn((bxc_on[:, 0, 0], bxc_on[0, :, 1]), byc_on, bxt, 
                                bounds_error=False, fill_value=byc_on.mean())
            yt_on.append(on_interp)

        yt = np.concatenate((np.stack(yt_off, axis=0), np.stack(yt_on, axis=0)), axis=-1)
        yt = torch.tensor(yt, dtype=torch.float32, device=device)
        yt = self.mix_layer(yt)
        return yt
        

class IdentityBaselineDecoder(nn.Module):

    def forward(self, z: torch.Tensor, *_) -> torch.Tensor:
        return z




        
