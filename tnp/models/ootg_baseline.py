from scipy.interpolate import LinearNDInterpolator
import numpy as np
import torch
from torch import nn

from ..data.on_off_grid import DataModality

class InterpBaselineEncoder(nn.Module):

    def forward(
        self, 
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        
        xc = used_modality.get(xc_on_grid, xc_off_grid)
        yc = used_modality.get(yc_on_grid, yc_off_grid)

        xc = xc.numpy(force=True) 
        yc = yc.numpy(force=True)
        xt = xt.numpy(force=True)

        yts = []
        for xc_b, yc_b, xt_b in zip(xc, yc, xt):
            interp = LinearNDInterpolator(xc_b, yc_b, fill_value=yc_b.mean())
            yts.append(interp(xt_b))
        yt = torch.tensor(np.stack(yts, axis=0), device=xc_on_grid.device)
        return yt
        

class IdentityBaselineDecoder(nn.Module):

    def forward(self, z: torch.Tensor, *_) -> torch.Tensor:
        return z




        
