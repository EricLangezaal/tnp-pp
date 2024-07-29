from typing import Tuple

import torch
from torch import nn

from ..utils.grids import coarsen_grid, flatten_grid
from ..data.on_off_grid import DataModality

class InterpBaselineEncoder(nn.Module):

    def __init__(
            self,
            interpolation_factors: Tuple[int,...],
    ):
        super().__init__()
        self.interpolation_factors = interpolation_factors

    def forward(
        self, 
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor,
        used_modality: DataModality = DataModality.BOTH,
    ) -> torch.Tensor:
        grid_locations = flatten_grid(coarsen_grid(xc_on_grid, self.interpolation_factors))
        grid_values = flatten_grid(coarsen_grid(yc_on_grid, self.interpolation_factors))

        B, U, Ydim = yc_off_grid.shape # 'U'nstructured
        S = grid_values.shape[-2] # 'S'tructured

        if used_modality == DataModality.ON_GRID:
            avg_grid = grid_values
        else:
            # from PT Encoder make the NN tensor
            nearest_idx = (
                (xc_off_grid[..., None, :] - grid_locations[:, None, ...]).abs().sum(dim=-1).argmin(dim=2)
            )
            s_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, S)
            u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
            s_range_idx = torch.arange(S).repeat(B, 1)
            u_range_idx = torch.arange(U).repeat(B, 1)
            nearest_mask = torch.zeros(B, U, S, dtype=torch.int, device=xc_off_grid.device)
            nearest_mask[u_batch_idx, u_range_idx, nearest_idx] = 1
            
            max_patch = nearest_mask.sum(dim=1).amax() + (used_modality == DataModality.BOTH)
            cumcount_idx = (nearest_mask.cumsum(dim=1) - 1)[u_batch_idx, u_range_idx, nearest_idx]

            joint_grid = torch.full((B, S, max_patch, Ydim), torch.nan, device=xc_off_grid.device)
            joint_grid[u_batch_idx, nearest_idx, cumcount_idx] = yc_off_grid
            if used_modality == DataModality.BOTH:
                joint_grid[s_batch_idx, s_range_idx, -1] = grid_values

            # for every coarsened location, average of grid points 
            # and all nearest off grid points which aren't masked out
            avg_grid = (joint_grid.nan_to_num()).sum(dim=2) / (~joint_grid.isnan()).sum(dim=2)

        nearest_idx_t = (
                (xt[..., None, :] - grid_locations[:, None, ...]).abs().sum(dim=-1).argmin(dim=2)
            )
        t_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, xt.shape[1])
        interp_result = avg_grid[t_batch_idx, nearest_idx_t]

        return interp_result
        

class IdentityBaselineDecoder(nn.Module):

    def forward(self, z: torch.Tensor, *_) -> torch.Tensor:
        return z




        
