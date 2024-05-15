import itertools
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import einops
import h5py
import numpy as np
import torch

from icicl.data.base import Batch, DataGenerator


@dataclass
class GriddedBatchWithInputs(Batch):
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    mc_grid: torch.Tensor
    mt_grid: torch.Tensor


class SmokeGenerator(DataGenerator):
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_grid_size: Tuple[int, int],
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: Optional[int] = None,
        max_num_trg: Optional[int] = None,
        renorm_batch: bool = False,
        return_as_gridded: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_grid_size = batch_grid_size
        self.min_num_ctx = torch.as_tensor(min_num_ctx, dtype=torch.int)
        self.max_num_ctx = torch.as_tensor(max_num_ctx, dtype=torch.int)

        if (min_num_trg is not None) and (max_num_trg is not None):
            self.min_num_trg = torch.as_tensor(min_num_trg, dtype=torch.int)
            self.max_num_trg = torch.as_tensor(max_num_trg, dtype=torch.int)
        else:
            self.min_num_trg = None
            self.max_num_trg = None

        self.renorm_batch = renorm_batch
        self.return_as_gridded = return_as_gridded
        self.dataset, self.spatial_range = self.load_data(data_dir, split)

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        """Generate batch of data.
        Returns:
            batch: Batch.
        """
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]
        # Sample number of context points
        nc = torch.randint(low=self.min_num_ctx, high=self.max_num_ctx + 1, size=())
        # If only evaluating on a certain number of target points
        if self.min_num_trg is not None:
            nt = torch.randint(low=self.min_num_trg, high=self.max_num_trg + 1, size=())
        else:
            nt = np.prod(self.batch_grid_size) - nc
        # Sample dataset idx.
        dataset_idxs = self.sample_dataset_idx(batch_size)
        # Sample batch grids from within datasets.
        batch_grids = self.sample_batch_grid(batch_size)
        # Get the data.
        batch = self.sample_batch(
            num_ctx=nc,
            num_trg=nt,
            dataset_idxs=dataset_idxs,
            batch_grids=batch_grids,
        )
        return batch

    def sample_dataset_idx(self, batch_size: int) -> List[int]:
        return [random.choice(range(len(self.dataset))) for _ in range(batch_size)]

    def sample_batch_grid(self, batch_size: int) -> List[Tuple[List, List]]:
        start_spatial_idx1 = random.choices(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[0]), k=batch_size
        )
        start_spatial_idx2 = random.choices(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[1]), k=batch_size
        )
        idx: List[Tuple[List, List]] = []
        for i, j in zip(start_spatial_idx1, start_spatial_idx2):
            idx.append(
                (
                    list(range(i, i + self.batch_grid_size[0])),
                    list(range(j, j + self.batch_grid_size[1])),
                )
            )
        return idx

    def sample_batch(
        self,
        num_ctx: int,
        num_trg: int,
        dataset_idxs: List[int],
        batch_grids: List[Tuple[List, List]],
    ) -> Batch:
        x_list, xc_list, xt_list = [], [], []
        y_list, yc_list, yt_list = [], [], []

        if self.return_as_gridded:
            x_grid_list, y_grid_list, mc_grid_list, mt_grid_list = [], [], [], []

        for dataset_idx, batch_grid in zip(dataset_idxs, batch_grids):
            batch_grid_product = np.array(list(itertools.product(*batch_grid)))
            y = self.dataset[dataset_idx][
                batch_grid_product[:, 0], batch_grid_product[:, 1]
            ]
            x = torch.stack(
                (
                    self.spatial_range[batch_grid_product[:, 0]],
                    self.spatial_range[batch_grid_product[:, 1]],
                ),
                dim=-1,
            )
            if self.renorm_batch:
                # Renormalise batch input locations.
                x = ((x - x.min()) / (x.max() - x.min()) * 2 - 1) * 3

            # Now randomly select context and target points.
            shuffled_idx = list(range(len(x)))
            random.shuffle(shuffled_idx)

            mc_idx = shuffled_idx[:num_ctx]
            mt_idx = shuffled_idx[num_ctx : (num_ctx + num_trg)]

            # Keep in order.
            mc_idx.sort()
            mt_idx.sort()

            if self.return_as_gridded:
                mc_grid_flat = torch.zeros(len(shuffled_idx), dtype=torch.bool)
                mc_grid_flat[mc_idx] = True
                mt_grid_flat = torch.zeros(len(shuffled_idx), dtype=torch.bool)
                mt_grid_flat[mt_idx] = True

                # Reshape into grid.
                x_grid = einops.rearrange(
                    x,
                    "(n1 n2) d -> n1 n2 d",
                    n1=len(batch_grid[0]),
                    n2=len(batch_grid[1]),
                )
                y_grid = einops.rearrange(
                    y,
                    "(n1 n2) d -> n1 n2 d",
                    n1=len(batch_grid[0]),
                    n2=len(batch_grid[1]),
                )
                mc_grid = einops.rearrange(
                    mc_grid_flat,
                    "(n1 n2) -> n1 n2",
                    n1=len(batch_grid[0]),
                    n2=len(batch_grid[1]),
                )
                mt_grid = einops.rearrange(
                    mt_grid_flat,
                    "(n1 n2) -> n1 n2",
                    n1=len(batch_grid[0]),
                    n2=len(batch_grid[1]),
                )

                x_grid_list.append(x_grid)
                y_grid_list.append(y_grid)
                mc_grid_list.append(mc_grid)
                mt_grid_list.append(mt_grid)

            x_list.append(x)
            y_list.append(y)
            xc_list.append(x[mc_idx])
            yc_list.append(y[mc_idx])
            xt_list.append(x[mt_idx])
            yt_list.append(y[mt_idx])

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        xc = torch.stack(xc_list)
        yc = torch.stack(yc_list)
        xt = torch.stack(xt_list)
        yt = torch.stack(yt_list)

        if self.return_as_gridded:
            x_grid = torch.stack(x_grid_list)
            y_grid = torch.stack(y_grid_list)
            mc_grid = torch.stack(mc_grid_list)
            mt_grid = torch.stack(mt_grid_list)

            return GriddedBatchWithInputs(
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
            )

        return Batch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
        )

    def normalise_data(
        self, data: torch.Tensor, data_dir: str, split: str
    ) -> torch.Tensor:
        if "train" not in split:
            print("Loading training data for normalising")
            fname = os.path.join(data_dir, "train_25000.h5")
            with h5py.File(fname, mode="r") as f:
                train_data = f["x"][:]

            print("Finished loading training data")
            train_data = torch.as_tensor(train_data, dtype=torch.float)
            data = (data - train_data.mean()) / (train_data.std())

        else:
            data = (data - data.mean()) / (data.std())

        return data

    def load_data(
        self, data_dir: str, split: str = "train"
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:

        print("Starting to load the data...")
        fname = os.path.join(data_dir, split + ".h5")
        with h5py.File(fname, mode="r") as f:
            data = f["x"][:]

        print("Data loaded...")
        data = torch.as_tensor(data, dtype=torch.float)
        data = self.normalise_data(data, data_dir, split)

        print("Mean data", data.mean(), "Std data", data.std())
        data = data.flatten(0, 1)

        # Expect final data shape [num_test, 2, x_grid, y_grid].
        data = torch.movedim(data, 1, -1)

        # Shuffle data.
        idx = torch.randperm(data.shape[0])
        data = data[idx].view(data.size())
        print("final data shape", data.shape)

        spatial_range = torch.linspace(-3, 3, 128)

        return data, spatial_range
