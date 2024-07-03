import math
import os
import random
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from .base import Batch, DataGenerator
from .on_off_grid import OOTGBatch
from ..utils.grids import flatten_grid, func_AvgPoolNd


@dataclass
class GriddedBatch(Batch):
    x_grid: torch.Tensor # grid of all data (batch, time, lat, lon, 3)
    y_grid: torch.Tensor # grid of all data (batch, time, lat, lon, 1), time optional

    mc_grid: torch.Tensor # boolean mask for context points in that grid
    mt_grid: torch.Tensor # boolean mask for target points in that grid
    m_grid: torch.Tensor # boolean mask for both (if some data is unused I think?, can be all True)


class BaseERA5DataGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        fnames: List[str],
        lat_range: Tuple[float, float] = (-90.0, 90.0),
        lon_range: Tuple[float, float] = (0.0 ,360.0),
        ref_date: str = "2000-01-01",
        data_vars: Tuple[str] = ("Tair",),
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load datasets.
        datasets = [
            xr.open_dataset(os.path.join(data_dir, fname))
            for fname in fnames  # pylint: disable=no-member
        ]

        # Merge datasets.
        dataset = xr.concat(datasets, "time")

        # Change time to hours since reference time.
        ref_datetime = datetime.strptime(ref_date, "%Y-%m-%d")
        ref_np_datetime = np.datetime64(ref_datetime)
        hours = (dataset["time"][:].data - ref_np_datetime) / np.timedelta64(1, "h")
        dataset = dataset.assign_coords(time=hours)

        # Apply specified lat/lon ranges.
        lat_idx = (dataset["latitude"][:] <= lat_range[1]) & (
            dataset["latitude"][:] >= lat_range[0]
        )
        lon_idx = (dataset["longitude"][:] <= lon_range[1]) & (
            dataset["longitude"][:] >= lon_range[0]
        )

        self.lat_range = lat_range
        self.lon_range = lon_range
        self.data_vars = data_vars
        self.all_input_vars = ["time", "lat", "lon"]
        self.data = {
            **{k: dataset[k][:, lat_idx, lon_idx] for k in data_vars},
            "time": dataset["time"][:],
            "lat": dataset["latitude"][lat_idx],
            "lon": dataset["longitude"][lon_idx],
        }


class ERA5DataGenerator(BaseERA5DataGenerator):
    def __init__(
        self,
        *,
        min_pc: float,
        max_pc: float,
        batch_grid_size: Tuple[int, int, int], #time, lat, lon
        max_nt: Optional[int] = None,
        min_num_total: int = 1,
        x_mean: Optional[Tuple[float, ...]] = None,
        x_std: Optional[Tuple[float, ...]] = None,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
        t_spacing: int = 1,
        use_time: bool = True,
        return_as_gridded: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pc_dist = torch.distributions.Uniform(min_pc, max_pc)

        # How large each sampled grid should be (in indicies).
        self.batch_grid_size = batch_grid_size
        self.max_nt = max_nt
        self.min_num_total = min_num_total

        self.t_spacing = t_spacing
        self.use_time = use_time
        if not use_time:
            assert (
                batch_grid_size[0] == 1
            ), "batch_grid_size[0] must be 1 if not using time."
            self.input_vars = ["lat", "lon"]
        else:
            self.input_vars = ["time", "lat", "lon"]

        # Assign means and stds.
        if x_mean is None or x_std is None:
            x_mean = tuple([self.data[k][:].mean().item() for k in self.input_vars])
            x_std = tuple([self.data[k][:].std().item() for k in self.input_vars])

        if y_mean is None or y_std is None:
            y_mean = tuple([self.data[k][:].mean().item() for k in self.data_vars])
            y_std = tuple([self.data[k][:].std().item() for k in self.data_vars])

        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float)
        self.y_std = torch.as_tensor(y_std, dtype=torch.float)

        self.return_as_gridded = return_as_gridded

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=batch_size)

        # Sample context proportion.
        pc = self.pc_dist.sample()

        # Get batch.
        batch = self.sample_batch(pc=pc, idxs=idxs)
        return batch
    
    def sample_idx(self, batch_size: int) -> List[Tuple[List, List, List]]:
        """Samples indices used to sample dataframe.

        Args:
            batch_size (int): Batch_size.

        Returns:
            Tuple[List, List, List]: Indicies.
        """
        # Keep sampling locations until one with enough non-missing values.
        # Must keep location the same across batch as missing values vary.
        valid_location = False
        while not valid_location:
            i = random.randint(0, len(self.data["lat"]) - self.batch_grid_size[1])
            lat_idx = list(range(i, i + self.batch_grid_size[1]))

            i = random.randint(0, len(self.data["lon"]) - self.batch_grid_size[2])
            lon_idx = list(range(i, i + self.batch_grid_size[2]))

            # Get number of non-missing points.
            num_points = self._get_num_points(lat_idx=lat_idx, lon_idx=lon_idx)
            num_points *= self.batch_grid_size[0]

            # Check if enough non-missing points to return batch.
            if num_points > self.min_num_total:
                valid_location = True

        time_idx: List[List] = []
        for _ in range(batch_size):
            i = random.randint(
                0, len(self.data["time"]) - 1 - self.t_spacing * self.batch_grid_size[0]
            )
            time_idx.append(
                list(
                    range(
                        i, i + self.t_spacing * self.batch_grid_size[0], self.t_spacing
                    )
                )
            )

        idx = [(time_idx[i], lat_idx, lon_idx) for i in range(len(time_idx))]
        return idx

    def sample_batch(self, pc: float, idxs: List[Tuple[List, ...]]) -> Batch:
        # Will build tensors from these later.
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        xcs: List[torch.Tensor] = []
        ycs: List[torch.Tensor] = []
        xts: List[torch.Tensor] = []
        yts: List[torch.Tensor] = []

        if self.return_as_gridded:
            x_grids, y_grids, mc_grids, mt_grids, m_grids = [], [], [], [], []

        # TODO: can we batch this?
        for idx in idxs:
            x_grid = torch.stack(
                torch.meshgrid(
                    *[
                        torch.as_tensor(self.data[k][idx[i]].data, dtype=torch.float)
                        for i, k in enumerate(self.all_input_vars)
                        if k in self.input_vars
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )

            y_grid = torch.stack(
                [
                    torch.as_tensor(self.data[k][idx].data, dtype=torch.float32)
                    for k in self.data_vars
                ],
                dim=-1,
            )

            if not self.use_time:
                y_grid = y_grid.squeeze(0)

            # Mask out data with any missing values.
            y_mask = torch.isnan(y_grid.sum(-1)).flatten()

            # Construct context and target mask.
            nc = math.ceil(pc * y_mask.numel())
            m_idx = torch.where(~y_mask)[0]
            m_idx = m_idx[torch.randperm(len(m_idx))]
            mc_idx = m_idx[:nc]
            if self.max_nt is None:
                mt_idx = m_idx[nc:]
            else:
                mt_idx = m_idx[nc : nc + self.max_nt]

            # Unravel into gridded form.
            mc_grid_idx = torch.unravel_index(mc_idx, y_grid.shape[:-1])
            mt_grid_idx = torch.unravel_index(mt_idx, y_grid.shape[:-1])
            m_grid_idx = torch.unravel_index(m_idx, y_grid.shape[:-1])

            # Normalise inputs and outputs.
            x_grid = (x_grid - self.x_mean) / self.x_std
            y_grid = (y_grid - self.y_mean) / self.y_std

            # Get flattened versions.
            x = x_grid[m_grid_idx]
            y = y_grid[m_grid_idx]
            xc = x_grid[mc_grid_idx]
            yc = y_grid[mc_grid_idx]
            xt = x_grid[mt_grid_idx]
            yt = y_grid[mt_grid_idx]

            xs.append(x)
            ys.append(y)
            xcs.append(xc)
            ycs.append(yc)
            xts.append(xt)
            yts.append(yt)

            if self.return_as_gridded:
                mc_grid = torch.zeros(y_grid.shape[:-1], dtype=torch.bool)
                mt_grid = torch.zeros(y_grid.shape[:-1], dtype=torch.bool)
                m_grid = torch.zeros(y_grid.shape[:-1], dtype=torch.bool)
                mc_grid[mc_grid_idx] = True
                mt_grid[mt_grid_idx] = True
                m_grid[m_grid_idx] = True

                # Fill nans with zeros. These will be masked out anyway!
                y_grid = torch.nan_to_num(y_grid, nan=-9999.99)

                x_grids.append(x_grid)
                y_grids.append(y_grid)
                mc_grids.append(mc_grid)
                mt_grids.append(mt_grid)
                m_grids.append(m_grid)

        x = torch.stack(xs)
        y = torch.stack(ys)
        xc = torch.stack(xcs)
        yc = torch.stack(ycs)
        xt = torch.stack(xts)
        yt = torch.stack(yts)

        if self.return_as_gridded:
            x_grid = torch.stack(x_grids)
            y_grid = torch.stack(y_grids)
            mc_grid = torch.stack(mc_grids)
            mt_grid = torch.stack(mt_grids)
            m_grid = torch.stack(m_grids)
            return GriddedBatch(
                x=x,
                y=y,
                xc=xc,
                yc=yc,
                xt=xt,
                yt=yt,
                x_grid=x_grid,
                y_grid=y_grid,
                mc_grid=mc_grid,
                mt_grid=mt_grid,
                m_grid=m_grid,
            )

        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)

    def _get_num_points(
        self,
        lat_idx: List[int],
        lon_idx: List[int],
        time_idx: Optional[List[int]] = None,
    ) -> int:
        time_idx = [0] if time_idx is None else time_idx

        y = torch.stack(
            [
                torch.as_tensor(
                    self.data[k][time_idx, lat_idx, lon_idx].data, dtype=torch.float32
                )
                for k in self.data_vars
            ],
            dim=-1,
        )
        y_mask = torch.isnan(y.sum(-1))
        return (~y_mask).sum()
    

class OOTGEra5DataGenerator(ERA5DataGenerator):

    def __init__(
            self,
            *,
            coarsen_factors: Union[Tuple[int, int], Tuple[int, int, int]],
            **kwargs,
    ):
        super().__init__(**kwargs)


        assert len(coarsen_factors) + (not self.use_time) == len(self.batch_grid_size),  (
            "please specify a coarsing for each grid dimension"
        )
        self.coarsen_factors = tuple(coarsen_factors)

    def sample_batch(self, pc: float, idxs: List[Tuple[List]]) -> OOTGBatch:
        era5_batch = super().sample_batch(pc, idxs)
        assert isinstance(era5_batch, GriddedBatch), "OOTG data requires setting return_as_gridded=True"

        xc_on_grid = coarsen_grid(era5_batch.x_grid, self.coarsen_factors)
        yc_on_grid = coarsen_grid(era5_batch.y_grid, self.coarsen_factors)

        xc = torch.cat((era5_batch.xc, flatten_grid(xc_on_grid)), dim=-2)
        yc = torch.cat((era5_batch.yc, flatten_grid(yc_on_grid)), dim=-2)

        # NOTE: order here is different from synthetic.
        x = torch.cat((xc, era5_batch.xt), dim=-2)
        y = torch.cat((yc, era5_batch.yt), dim=-2)

        return OOTGBatch(
           x=x,
           y=y,
           xc=xc,
           yc=yc,
           xt=era5_batch.xt,
           yt=era5_batch.yt,
           xc_on_grid=xc_on_grid,
           yc_on_grid=yc_on_grid,
           xc_off_grid=era5_batch.xc,
           yc_off_grid=era5_batch.yc,
           gt_pred=None
        )

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