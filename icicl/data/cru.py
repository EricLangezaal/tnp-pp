import math
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import einops
import numpy as np
import torch
import xarray as xr

from .base import Batch, DataGenerator


class CRUDataGenerator(DataGenerator):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        data_dir: str,
        fnames: List[str],
        min_prop_ctx: float,
        max_prop_ctx: float,
        batch_grid_size: Tuple[int, int, int],
        lat_range: Tuple[float, float] = (-89.75, 89.75),
        lon_range: Tuple[float, float] = (-179.75, 179.75),
        max_num_total: Optional[int] = None,
        min_num_total: int = 1,
        x_mean: Optional[Tuple[float, float, float]] = None,
        x_std: Optional[Tuple[float, float, float]] = None,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
        ref_date: str = "2000-01-01",
        t_spacing: int = 1,
    ):
        super().__init__(samples_per_epoch=samples_per_epoch, batch_size=batch_size)

        self.prop_ctx_dist = torch.distributions.Uniform(min_prop_ctx, max_prop_ctx)

        # How large each sampled grid should be (in indicies).
        self.batch_grid_size = batch_grid_size
        self.dim = np.prod(batch_grid_size)
        self.max_num_total = max_num_total
        self.min_num_total = min_num_total

        # Store ranges for plotting.
        self.lat_range = lat_range
        self.lon_range = lon_range

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
        lon_idx = (dataset["lon"][:] <= lon_range[1]) & (
            dataset["lon"][:] >= lon_range[0]
        )
        lat_idx = (dataset["lat"][:] <= lat_range[1]) & (
            dataset["lat"][:] >= lat_range[0]
        )

        self.data = {
            "Tair": dataset["Tair"][:, lat_idx, lon_idx],
            "time": dataset["time"][:][::t_spacing],
            "lat": dataset["lat"][lat_idx],
            "lon": dataset["lon"][lon_idx],
        }

        # Assign means and stds.
        if x_mean is None or x_std is None:
            x_mean = [self.data[k][:].mean().item() for k in ["time", "lat", "lon"]]
            x_std = [self.data[k][:].std().item() for k in ["time", "lat", "lon"]]

        if y_mean is None or y_std is None:
            y_mean = self.data["Tair"].mean().item()
            y_std = self.data["Tair"].std().item()

        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float)
        self.y_std = torch.as_tensor(y_std, dtype=torch.float)

    def generate_batch(self) -> Batch:
        # (batch_size, n, 3).
        idxs = self.sample_idx(batch_size=self.batch_size)

        # Sample context proportion.
        pc = self.prop_ctx_dist.sample()

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
            i = random.randint(0, len(self.data["lon"]) - 1 - self.batch_grid_size[2])
            lon_idx = list(range(i, i + self.batch_grid_size[1]))

            i = random.randint(0, len(self.data["lat"]) - 1 - self.batch_grid_size[1])
            lat_idx = list(range(i, i + self.batch_grid_size[2]))

            # Get number of non-missing points.
            num_points = self._get_num_points(lat_idx=lat_idx, lon_idx=lon_idx)
            num_points *= self.batch_grid_size[0]
            if num_points > self.min_num_total:
                valid_location = True

        time_idx: List[List] = []
        for _ in range(batch_size):
            i = random.randint(0, len(self.data["time"]) - 1 - self.batch_grid_size[0])
            time_idx.append(list(range(i, i + self.batch_grid_size[0])))

        idx = [(time_idx[i], lat_idx, lon_idx) for i in range(len(time_idx))]
        return idx

    def sample_batch(self, pc: float, idxs: List[Tuple[List, List, List]]) -> Batch:
        # Will build tensors from these later.
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        xcs: List[torch.Tensor] = []
        ycs: List[torch.Tensor] = []
        xts: List[torch.Tensor] = []
        yts: List[torch.Tensor] = []

        for idx in idxs:
            # Crackers, but fast.
            idx_grid = self._create_idx_grid(idx)

            x = torch.stack(
                [
                    torch.as_tensor(
                        self.data["time"][idx_grid[:, 0]].data, dtype=torch.float
                    ),
                    torch.as_tensor(
                        self.data["lat"][idx_grid[:, 1]].data, dtype=torch.float
                    ),
                    torch.as_tensor(
                        self.data["lon"][idx_grid[:, 2]].data, dtype=torch.float
                    ),
                ],
                dim=-1,
            )

            y_raw = self.data["Tair"][idx[0], idx[1], idx[2]]
            y_mask = np.isnan(y_raw.data)

            y = (
                torch.as_tensor(y_raw.data[~y_mask], dtype=torch.float32).unsqueeze(-1)
                - self.y_mean
            ) / self.y_std
            x = (x[~y_mask.flatten()] - self.x_mean) / self.x_std

            # Sample indices for context / target.
            shuffled_idx = np.arange(len(y))
            np.random.shuffle(shuffled_idx)
            shuffled_idx = shuffled_idx[: self.max_num_total]
            x = x[shuffled_idx]
            y = y[shuffled_idx]

            num_ctx = math.ceil(pc * len(y))

            xc = x[:num_ctx]
            yc = y[:num_ctx]
            xt = x[num_ctx:]
            yt = y[num_ctx:]

            xs.append(x)
            ys.append(y)
            xcs.append(xc)
            ycs.append(yc)
            xts.append(xt)
            yts.append(yt)

        x = torch.stack(xs)
        y = torch.stack(ys)
        xc = torch.stack(xcs)
        yc = torch.stack(ycs)
        xt = torch.stack(xts)
        yt = torch.stack(yts)

        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)

    def _create_idx_grid(self, idx: Tuple[List, List, List]) -> torch.Tensor:
        """Constructs a grid of tensors from given list of indices.

        Args:
            idx (Tuple[List, List, List]): List of indicies.

        Returns:
            torch.Tensor: Grid of tensors.
        """
        idx_grid = torch.stack(
            torch.meshgrid(
                torch.as_tensor(idx[0]),
                torch.as_tensor(idx[1]),
                torch.as_tensor(idx[2]),
            ),
            dim=-1,
        )
        idx_grid = einops.rearrange(idx_grid, "n1 n2 n3 d -> (n1 n2 n3) d")

        return idx_grid

    def _get_num_points(
        self,
        lat_idx: List[int],
        lon_idx: List[int],
        time_idx: Optional[List[int]] = None,
    ) -> int:
        time_idx = [0] if time_idx is None else time_idx

        y = self.data["Tair"][time_idx, lat_idx, lon_idx]
        y_mask = np.isnan(y.data)
        return (~y_mask).sum()
