from scipy.interpolate import griddata, interpn
import numpy as np
import torch
from torch import nn
import warnings

from ..networks.mlp import MLP
from ..data.on_off_grid import DataModality

class InterpBaselineEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.mix_prop_logit = torch.nn.Parameter(torch.tensor(0.0))

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
            off_interp = griddata(bxc_off, byc_off, bxt, fill_value=byc_off.mean())
            yt_off.append(off_interp)

            on_interp = interpn((bxc_on[:, 0, 0], bxc_on[0, :, 1]), byc_on, bxt, 
                                bounds_error=False, fill_value=byc_on.mean())
            yt_on.append(on_interp)

        yt_off = torch.tensor(np.stack(yt_off, axis=0), dtype=torch.float32, device=device)
        yt_on = torch.tensor(np.stack(yt_on, axis=0), dtype=torch.float32, device=device)
        mix_prop = nn.functional.sigmoid(self.mix_prop_logit)

        yt = mix_prop * yt_off + (1 - mix_prop) * yt_on
        return yt
        

class IdentityBaselineDecoder(nn.Module):

    def forward(self, z: torch.Tensor, *_) -> torch.Tensor:
        return z




        
