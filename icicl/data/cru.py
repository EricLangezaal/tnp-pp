import math
import random
from typing import List, Optional, Tuple

import einops
import netCDF4
import numpy as np
import torch

from .base import Batch, DataGenerator


class CRUDataGenerator(DataGenerator):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        fname: str,
        min_prop_ctx: float,
        max_prop_ctx: float,
        batch_grid_size: Tuple[int, int, int],
        lat_range: Tuple[float, float] = (-89.75, 89.75),
        lon_range: Tuple[float, float] = (-179.75, 179.75),
        max_num_total: Optional[int] = None,
    ):
        super().__init__(samples_per_epoch=samples_per_epoch, batch_size=batch_size)

        self.prop_ctx_dist = torch.distributions.Uniform(min_prop_ctx, max_prop_ctx)

        # How large each sampled grid should be (in indicies).
        self.batch_grid_size = batch_grid_size
        self.dim = np.prod(batch_grid_size)
        self.max_num_total = max_num_total

        # Load dataset.
        dataset = netCDF4.Dataset(fname, "r")

        # Apply specified lat/lon ranges.
        lon_idx = (dataset["lon"][:] <= lon_range[1]) & (
            dataset["lon"][:] >= lon_range[0]
        )
        lat_idx = (dataset["lat"][:] <= lat_range[1]) & (
            dataset["lat"][:] >= lat_range[0]
        )

        self.data = {
            "Tair": dataset["Tair"][:, lat_idx, lon_idx],
            "time": dataset["time"][:],
            "lat": dataset["lat"][lat_idx],
            "lon": dataset["lon"][lon_idx],
        }

        self.x_mean = torch.as_tensor(
            [self.data[k].data.mean() for k in ["time", "lat", "lon"]],
            dtype=torch.float,
        )
        self.x_std = torch.as_tensor(
            [self.data[k].data.std() for k in ["time", "lat", "lon"]],
            dtype=torch.float,
        )
        self.y_mean = torch.as_tensor(
            self.data["Tair"].data[~self.data["Tair"].mask].mean(),
            dtype=torch.float,
        )
        self.y_std = torch.as_tensor(
            self.data["Tair"].data[~self.data["Tair"].mask].std(),
            dtype=torch.float,
        )

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
        # Must keep location the same across batch as missing values vary.
        i = random.randint(0, len(self.data["lon"]) - 1 - self.batch_grid_size[2])
        lon_idx = list(range(i, i + self.batch_grid_size[1]))

        i = random.randint(0, len(self.data["lat"]) - 1 - self.batch_grid_size[1])
        lat_idx = list(range(i, i + self.batch_grid_size[2]))

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
            idx_grid = torch.stack(
                torch.meshgrid(
                    torch.as_tensor(idx[0]),
                    torch.as_tensor(idx[1]),
                    torch.as_tensor(idx[2]),
                ),
                dim=-1,
            )
            idx_grid = einops.rearrange(idx_grid, "n1 n2 n3 d -> (n1 n2 n3) d")

            x = torch.stack(
                [
                    torch.as_tensor(
                        self.data["time"][idx_grid[:, 0]].data, dtype=float
                    ),
                    torch.as_tensor(self.data["lat"][idx_grid[:, 1]].data, dtype=float),
                    torch.as_tensor(self.data["lon"][idx_grid[:, 2]].data, dtype=float),
                ],
                dim=-1,
            )

            y_raw = self.data["Tair"][
                idx_grid[:, 0].tolist(),
                idx_grid[:, 1].tolist(),
                idx_grid[:, 2].tolist(),
            ]

            y_mask = y_raw.mask
            y = (
                torch.as_tensor(y_raw.data[~y_mask]).unsqueeze(-1) - self.y_mean
            ) / self.y_std
            x = (x[~y_mask.flatten()] - self.x_mean) / self.x_std

            # Sample indices for context / target.
            shuffled_idx = np.arange(len(y))
            np.random.shuffle(shuffled_idx)

            num_ctx = math.ceil(pc * len(y))
            num_total = (
                -1 if self.max_num_total is None else min(self.max_num_total, len(y))
            )

            xc = x[shuffled_idx[:num_ctx]]
            yc = y[shuffled_idx[:num_ctx]]
            xt = x[shuffled_idx[num_ctx:num_total]]
            yt = y[shuffled_idx[num_ctx:num_total]]

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
