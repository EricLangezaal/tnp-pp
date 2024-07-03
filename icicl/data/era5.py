import itertools
import math
import os
import random
import warnings
from abc import ABC
from dataclasses import dataclass
from functools import partial
from datetime import datetime
from typing import List, Optional, Tuple, Union

import dask
import numpy as np
import torch
import xarray as xr

from .base import Batch, DataGenerator
from .on_off_grid import OOTGBatch
from ..utils.grids import flatten_grid, func_AvgPoolNd


@dataclass
class GriddedBatch(Batch):
    x_grid: torch.Tensor
    y_grid: torch.Tensor


class BaseERA5DataGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        fnames: List[str],
        lat_range: Tuple[float, float] = (90.0, -90.0),
        lon_range: Tuple[float, float] = (0.0, 360.0),
        batch_grid_size: Tuple[int, int, int],
        min_num_batch: int = 1,
        ref_date: str = "2000-01-01",
        data_vars: Tuple[str] = ("t2m",),
        t_spacing: int = 1,
        use_time: bool = True,
        x_mean: Optional[Tuple[float, ...]] = None,
        x_std: Optional[Tuple[float, ...]] = None,
        y_mean: Optional[Tuple[float, ...]] = None,
        y_std: Optional[Tuple[float, ...]] = None,
        lazy_loading: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        def _preprocess(
            ds: xr.Dataset,
            lat_range: Tuple[float, float],
            lon_range: Tuple[float, float],
        ):
            return ds.sel(
                latitude=slice(*lat_range),
                longitude=slice(*lon_range),
            )

        # Load datasets.
        dataset = xr.open_mfdataset(
            [os.path.join(data_dir, fname) for fname in fnames],
            preprocess=partial(_preprocess, lat_range=lat_range, lon_range=lon_range),
            chunks="auto",
        )

        drop_vars = list(set(dataset.data_vars) - set(data_vars))
        dataset = dataset.drop_vars(drop_vars).astype(np.float32)

        # Change time to hours since reference time.
        ref_datetime = datetime.strptime(ref_date, "%Y-%m-%d")
        ref_np_datetime = np.datetime64(ref_datetime)
        hours = (dataset["time"][:].data - ref_np_datetime) / np.timedelta64(1, "h")
        dataset = dataset.assign_coords(time=hours)

        self.lat_range = lat_range
        self.lon_range = lon_range
        self.data_vars = data_vars
        self.all_input_vars = ["time", "latitude", "longitude"]

        self.lazy_loading = lazy_loading
        self.data = {
            **{k: dataset[k] for k in data_vars},
            "time": dataset["time"],
            "latitude": dataset["latitude"],
            "longitude": dataset["longitude"],
        }
        if not lazy_loading:
            self.data = dask.compute(self.data)[0]

        # How large each sampled grid should be (in indicies).
        self.batch_grid_size = batch_grid_size
        self.min_num_batch = min_num_batch

        self.t_spacing = t_spacing
        self.use_time = use_time
        if not use_time:
            assert (
                batch_grid_size[0] == 1
            ), "batch_grid_size[0] must be 1 if not using time."
            self.input_vars = ["latitude", "longitude"]
        else:
            self.input_vars = ["time", "latitude", "longitude"]

        # Assign means and stds.
        if x_mean is None or x_std is None:
            x_mean = tuple(self.data[k][:].mean().item() for k in self.input_vars)
            x_std = tuple(self.data[k][:].std().item() for k in self.input_vars)

        if y_mean is None or y_std is None:
            warnings.warn("Computing mean and standard deviation of observations.")
            y_mean = tuple(self.data[k][:].mean().values.item() for k in self.data_vars)
            y_std = tuple(self.data[k][:].std().values.item() for k in self.data_vars)

        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float)
        self.y_std = torch.as_tensor(y_std, dtype=torch.float)

    def sample_idx(self, batch_size: int) -> List[Tuple[List, List, List]]:
        """Samples indices used to sample dataframe.

        Args:
            batch_size (int): Batch_size.

        Returns:
            Tuple[List, List, List]: Indicies.
        """
        # Keep sampling locations until one with enough non-missing values.
        # Must keep location the same across batch as missing values vary.
        if self.batch_grid_size[1] > len(self.data["latitude"]) or (
            self.batch_grid_size[2] > len(self.data["longitude"])
        ):
            raise ValueError("Grid size is too large!")

        i = random.randint(0, len(self.data["latitude"]) - self.batch_grid_size[1])
        lat_idx = list(range(i, i + self.batch_grid_size[1]))

        i = random.randint(0, len(self.data["longitude"]) - self.batch_grid_size[2])
        lon_idx = list(range(i, i + self.batch_grid_size[2]))

        time_idx: List[List] = []
        for _ in range(batch_size):
            i = random.randint(
                0,
                len(self.data["time"]) - self.t_spacing * self.batch_grid_size[0],
            )
            time_idx.append(
                list(
                    range(
                        i,
                        i + self.t_spacing * self.batch_grid_size[0],
                        self.t_spacing,
                    )
                )
            )

        idx = [(time_idx[i], lat_idx, lon_idx) for i in range(len(time_idx))]
        return idx


class ERA5DataGenerator(BaseERA5DataGenerator):
    def __init__(
        self,
        *,
        min_pc: float,
        max_pc: float,
        max_nt: Optional[int] = None,
        return_grid: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pc_dist = torch.distributions.Uniform(min_pc, max_pc)

        # How large each sampled grid should be (in indicies).
        self.max_nt = max_nt

        # Whether to return the grid.
        self.return_grid = return_grid

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Sample context proportion.
        pc = self.pc_dist.sample()

        # Get batch.
        batch = self.sample_batch(pc=pc, idxs=idxs)
        return batch

    def sample_batch(self, pc: float, idxs: List[Tuple[List, ...]]) -> Batch:
        x_grid = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        *[
                            torch.as_tensor(
                                self.data[k][idx[i]].data, dtype=torch.float
                            )
                            for i, k in enumerate(self.all_input_vars)
                            if k in self.input_vars
                        ],
                        indexing="ij",
                    ),
                    dim=-1,
                )
                for idx in idxs
            ],
            dim=0,
        )

        y_grid_list = [[self.data[k][idx].data for k in self.data_vars] for idx in idxs]

        if self.lazy_loading:
            y_grid_list = dask.compute(y_grid_list)[0]

        y_grid = torch.stack(
            [
                torch.stack(
                    [torch.as_tensor(y, dtype=torch.float32) for y in y_grid_], dim=-1
                )
                for y_grid_ in y_grid_list
            ],
            dim=0,
        )

        if not self.use_time:
            y_grid = y_grid.squeeze(1)

        # Normalise inputs and outputs.
        x_grid = (x_grid - self.x_mean) / self.x_std
        y_grid = (y_grid - self.y_mean) / self.y_std

        # Assumes same masking pattern for each grid.
        y_mask = torch.isnan(y_grid[0].sum(-1)).flatten()
        nc = math.ceil(pc * (~y_mask).sum())
        m_idx_list = [torch.where(~y_mask)[0] for _ in range(len(idxs))]
        m_idx = torch.stack(
            [m_idx_[torch.randperm(len(m_idx_))] for m_idx_ in m_idx_list], dim=0
        )
        mc_idx = m_idx[:, :nc]
        if self.max_nt is None:
            mt_idx = m_idx[:, nc:]
        else:
            mt_idx = m_idx[:, nc : nc + self.max_nt]

        # Unravel into gridded form.
        mc_grid_idx = torch.unravel_index(mc_idx, y_grid.shape[1:-1])
        mt_grid_idx = torch.unravel_index(mt_idx, y_grid.shape[1:-1])
        m_grid_idx = torch.unravel_index(m_idx, y_grid.shape[1:-1])
        batch_idx = torch.arange(len(idxs)).unsqueeze(-1)

        # Get flattened versions.
        x = x_grid[(batch_idx,) + m_grid_idx]
        y = y_grid[(batch_idx,) + m_grid_idx]
        xc = x_grid[(batch_idx,) + mc_grid_idx]
        yc = y_grid[(batch_idx,) + mc_grid_idx]
        xt = x_grid[(batch_idx,) + mt_grid_idx]
        yt = y_grid[(batch_idx,) + mt_grid_idx]

        if self.return_grid:
            return GriddedBatch(
                x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt, x_grid=x_grid, y_grid=y_grid
            )

        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)


class ERA5DataGeneratorFRF(ERA5DataGenerator):
    def __init__(
        self,
        *,
        receptive_field: Tuple[int, ...],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.receptive_field = receptive_field

    def sample_batch(self, pc: float, idxs: List[Tuple[List, ...]]) -> Batch:
        x_grid = torch.stack(
            [
                torch.stack(
                    torch.meshgrid(
                        *[
                            torch.as_tensor(
                                self.data[k][idx[i]].data, dtype=torch.float
                            )
                            for i, k in enumerate(self.all_input_vars)
                            if k in self.input_vars
                        ],
                        indexing="ij",
                    ),
                    dim=-1,
                )
                for idx in idxs
            ],
            dim=0,
        )

        y_grid_list = [[self.data[k][idx].data for k in self.data_vars] for idx in idxs]

        if self.lazy_loading:
            y_grid_list = dask.compute(y_grid_list)[0]

        y_grid = torch.stack(
            [
                torch.stack(
                    [torch.as_tensor(y, dtype=torch.float32) for y in y_grid_], dim=-1
                )
                for y_grid_ in y_grid_list
            ],
            dim=0,
        )

        if not self.use_time:
            y_grid = y_grid.squeeze(1)

        # Normalise inputs and outputs.
        x_grid = (x_grid - self.x_mean) / self.x_std
        y_grid = (y_grid - self.y_mean) / self.y_std

        # Construct context mask using receptive field.
        inner_slice = tuple(slice(rf, -rf) for rf in self.receptive_field)
        outer_slices = [(slice(0, rf), slice(-rf, None)) for rf in self.receptive_field]

        # Assumes same masking pattern for each grid.
        y_mask = torch.isnan(y_grid[0].sum(-1))
        y_mask_inner = y_mask.clone()
        y_mask_outer = y_mask.clone()
        y_mask_outer[inner_slice] = True

        for outer_slice in itertools.product(*outer_slices):
            y_mask_inner[outer_slice] = True

        m_idx = torch.where((~y_mask).flatten())[0]
        m_idx_inner = torch.where((~y_mask_inner).flatten())[0]
        m_idx_outer = torch.where((~y_mask_outer).flatten())[0]

        m_idx = torch.stack(
            [m_idx[torch.randperm(len(m_idx))] for _ in range(len(idxs))], dim=0
        )
        m_idx_inner = torch.stack(
            [m_idx_inner[torch.randperm(len(m_idx_inner))] for _ in range(len(idxs))],
            dim=0,
        )
        m_idx_outer = torch.stack(
            [m_idx_outer[torch.randperm(len(m_idx_outer))] for _ in range(len(idxs))],
            dim=0,
        )

        # Construct context and target mask.
        nc_inner = math.ceil(pc * len(m_idx_inner[0]))
        nc_outer = math.ceil(pc * len(m_idx_outer[0]))
        mc_idx = torch.cat(
            (m_idx_inner[:, :nc_inner], m_idx_outer[:, :nc_outer]), dim=1
        )
        mt_idx = m_idx_inner[:, nc_inner:]

        if self.max_nt is not None:
            mt_idx = mt_idx[:, : self.max_nt]

        # Unravel into gridded form.
        mc_grid_idx = torch.unravel_index(mc_idx, y_grid.shape[1:-1])
        mt_grid_idx = torch.unravel_index(mt_idx, y_grid.shape[1:-1])
        m_grid_idx = torch.unravel_index(m_idx, y_grid.shape[1:-1])
        batch_idx = torch.arange(len(idxs)).unsqueeze(-1)

        # Get flattened versions.
        x = x_grid[(batch_idx,) + m_grid_idx]
        y = y_grid[(batch_idx,) + m_grid_idx]
        xc = x_grid[(batch_idx,) + mc_grid_idx]
        yc = y_grid[(batch_idx,) + mc_grid_idx]
        xt = x_grid[(batch_idx,) + mt_grid_idx]
        yt = y_grid[(batch_idx,) + mt_grid_idx]

        if self.return_grid:
            return GriddedBatch(
                x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt, x_grid=x_grid, y_grid=y_grid
            )

        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)
    

class ERA5OOTGDataGenerator(ERA5DataGenerator):
    
    def __init__(
        self,
        coarsen_factors: Tuple[int, ...] = (4, 4),
        **kwargs,
    ):
        kwargs["return_grid"] = True
        super().__init__(**kwargs)

        assert len(coarsen_factors) + (not self.use_time) == len(self.batch_grid_size),  (
            "please specify a coarsing for each grid dimension"
        )
        self.coarsen_factors = tuple(coarsen_factors)

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        batch = super().generate_batch(batch_shape)
        assert isinstance(batch, GriddedBatch), "batch must be gridded."

        xc_on_grid = coarsen_grid(batch.x_grid, self.coarsen_factors)
        yc_on_grid = coarsen_grid(batch.y_grid, self.coarsen_factors) 
        
        xc = torch.cat((batch.xc, flatten_grid(xc_on_grid)), dim=-2)
        yc = torch.cat((batch.yc, flatten_grid(yc_on_grid)), dim=-2)
        # NOTE: order here is different from synthetic.
        x = torch.cat((xc, batch.xt), dim=-2)
        y = torch.cat((yc, batch.yt), dim=-2)

        return OOTGBatch(
           x=x,
           y=y,
           xc=xc,
           yc=yc,
           xt=batch.xt,
           yt=batch.yt,
           xc_on_grid=xc_on_grid,
           yc_on_grid=yc_on_grid,
           xc_off_grid=batch.xc,
           yc_off_grid=batch.yc,
           gt_pred=None
        )

class ERA5OOTGDataGeneratorFRF(ERA5OOTGDataGenerator, ERA5DataGeneratorFRF):
    # Confirmed this actually works.
    pass

def coarsen_grid(grid: torch.Tensor, coarsen_factors: Union[Tuple[int, int], Tuple[int, int, int]]) -> torch.Tensor:
    grid = grid.movedim(-1, 1) # move data dim to channel location

    coarse_grid = func_AvgPoolNd(
        n=grid.ndim - 2, 
        input=grid,
        kernel_size=coarsen_factors, 
        stride=coarsen_factors
    )
    coarse_grid = coarse_grid.movedim(1, -1) # move embed dim back to the end
    return coarse_grid