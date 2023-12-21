from typing import List, Tuple

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
        same_location_each_batch: bool = True,
    ):
        super().__init__(samples_per_epoch=samples_per_epoch, batch_size=batch_size)

        self.min_prop_ctx = min_prop_ctx
        self.max_prop_ctx = max_prop_ctx
        self.lat_range = lat_range
        self.lon_range = lon_range

        # How large each sampled grid should be (in indicies).
        self.batch_grid_size = batch_grid_size
        self.dim = np.prod(batch_grid_size)

        # Whether or not to use the same lat/lon each batch.
        # The issue with using different lat/lon per batch is masking discrepencies.
        self.same_location_each_batch = same_location_each_batch

        # Load dataset.
        self.dataset = netCDF4.Dataset(fname, "r")

        # Get corresponding indices for specified min/max lat/lon.
        min_lat_idx = np.argwhere(
            self.dataset.variables["lat"][:].data == lat_range[0]
        )[0][0]
        max_lat_idx = np.argwhere(
            self.dataset.variables["lat"][:].data == lat_range[1]
        )[0][0]
        min_lon_idx = np.argwhere(
            self.dataset.variables["lon"][:].data == lon_range[0]
        )[0][0]
        max_lon_idx = np.argwhere(
            self.dataset.variables["lon"][:].data == lon_range[1]
        )[0][0]
        self.dataset_idx_range = (
            (0, len(self.dataset.variables["time"])),
            (min_lat_idx, max_lat_idx),
            (min_lon_idx, max_lon_idx),
        )

        self.raw_dataset = {k: v[:] for k, v in self.dataset.variables.items()}

        self.x_mean = torch.as_tensor(
            [self.raw_dataset[k].data.mean() for k in ["time", "lat", "lon"]],
            dtype=torch.float,
        )
        self.x_std = torch.as_tensor(
            [self.raw_dataset[k].data.std() for k in ["time", "lat", "lon"]],
            dtype=torch.float,
        )
        self.y_mean = torch.as_tensor(
            self.raw_dataset["Tair"].data[~self.raw_dataset["Tair"].mask].mean(),
            dtype=torch.float,
        )
        self.y_std = torch.as_tensor(
            self.raw_dataset["Tair"].data[~self.raw_dataset["Tair"].mask].std(),
            dtype=torch.float,
        )

    def generate_batch(self) -> Batch:
        # (batch_size, n, 3).
        grids = self.sample_grids(batch_size=self.batch_size)

        # Get air temperature!
        tair = [
            self.raw_dataset["Tair"][
                grids[i, :, 0].tolist(),
                grids[i, :, 1].tolist(),
                grids[i, :, 2].tolist(),
            ]
            for i in range(self.batch_size)
        ]

        if any([tair[i].count() == 0 for i in range(len(tair))]):
            raise ValueError("Please no.")

        # Make sure max pc does not exceed proportion of non-missing values.
        max_pc = max(tair[i].mask.sum() / tair[i].size for i in range(len(tair)))
        pc = self.sample_prop(self.min_prop_ctx, min(self.max_prop_ctx, max_pc))

        batch = self.sample_batch(pc=pc, tair=tair, grids=grids)
        return batch

    def sample_grids(self, batch_size: int) -> torch.Tensor:
        """Samples indices of input slices, then meshes them together to for grid.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Input grid from which to get data.
        """

        # First sample indices of the input slices.
        def sample_k_in_n(k: int, min_idx: int, max_idx: int):
            assert k <= (max_idx - min_idx)
            start_idx = torch.randint(min_idx, max_idx - k, size=())
            return torch.arange(start=start_idx, end=start_idx + k, step=1)

        if not self.same_location_each_batch:
            # Sample lat/lon for each element.
            idxs = [
                [
                    sample_k_in_n(k, min_idx, max_idx)
                    for k, (min_idx, max_idx) in zip(
                        self.batch_grid_size, self.dataset_idx_range
                    )
                ]
                for _ in range(batch_size)
            ]
        else:
            # Use same lat/lon for each element.
            lat_lon_idxs = [
                sample_k_in_n(k, min_idx, max_idx)
                for k, (min_idx, max_idx) in zip(
                    self.batch_grid_size[1:], self.dataset_idx_range[1:]
                )
            ]
            idxs = [
                [
                    sample_k_in_n(
                        self.batch_grid_size[0],
                        self.dataset_idx_range[0][0],
                        self.dataset_idx_range[0][1],
                    ),
                    *lat_lon_idxs,
                ]
                for _ in range(batch_size)
            ]

        # Convert indices into stacked list of 3-D coordinates.
        def create_meshgrid(idxs: List[torch.Tensor]):
            return torch.stack(
                torch.meshgrid(*idxs),
                dim=-1,
            )

        grids = torch.stack([create_meshgrid(idx) for idx in idxs])
        grids = einops.rearrange(grids, "m n1 n2 n3 d -> m (n1 n2 n3) d")

        return grids

    def sample_prop(self, min_prop: float, max_prop: float) -> torch.Tensor:
        # Sample proportions to mask.
        prop = torch.rand(size=()) * (max_prop - min_prop) + min_prop

        return prop

    def sample_batch(self, pc: float, tair: List, grids: torch.Tensor) -> Batch:
        # Extract time/lat/lon values from indices.
        x_grid = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(
                            self.raw_dataset["time"][grids[i, :, 0]].data,
                            dtype=torch.float,
                        ),
                        torch.as_tensor(
                            self.raw_dataset["lat"][grids[i, :, 1]].data,
                            dtype=torch.float,
                        ),
                        torch.as_tensor(
                            self.raw_dataset["lon"][grids[i, :, 2]].data,
                            dtype=torch.float,
                        ),
                    ],
                    dim=-1,
                )
                for i in range(len(grids))
            ]
        )

        x_grid = (x_grid - self.x_mean) / self.x_std

        all_tair: List[torch.Tensor] = []
        all_grids: List[torch.Tensor] = []
        ctx_grids: List[torch.Tensor] = []
        ctx_tair: List[torch.Tensor] = []
        trg_grids: List[torch.Tensor] = []
        trg_tair: List[torch.Tensor] = []
        for i, batch_tair in enumerate(tair):
            # Identify non-masked data.
            if isinstance(batch_tair.mask, np.bool_):
                assert batch_tair.mask is False
                data_idx = np.arange(batch_tair.size)
            else:
                data_idx = np.arange(batch_tair.size)[~batch_tair.mask]

            prop_ctx = pc / (len(data_idx) / batch_tair.size)

            shuffled = np.arange(len(data_idx))
            np.random.shuffle(shuffled)

            ctx_idx = data_idx[shuffled[: int(prop_ctx * len(data_idx))]]
            trg_idx = data_idx[shuffled[int(prop_ctx * len(data_idx)) :]]

            batch_tair_data = (
                torch.as_tensor(batch_tair.data.flatten(), dtype=torch.float).unsqueeze(
                    -1
                )
                - self.y_mean
            ) / self.y_std

            all_tair.append(batch_tair_data[data_idx, ...])
            all_grids.append(x_grid[i, data_idx, ...])
            ctx_grids.append(x_grid[i, ctx_idx, ...])
            ctx_tair.append(batch_tair_data[ctx_idx, ...])
            trg_grids.append(x_grid[i, trg_idx, ...])
            trg_tair.append(batch_tair_data[trg_idx, ...])

        x = torch.stack(all_grids, dim=0)
        y = torch.stack(all_tair, dim=0)
        xc = torch.stack(ctx_grids, dim=0)
        yc = torch.stack(ctx_tair, dim=0)
        xt = torch.stack(trg_grids, dim=0)
        yt = torch.stack(trg_tair, dim=0)

        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)
