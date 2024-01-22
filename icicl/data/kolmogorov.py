import itertools
import os
import random
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import Batch, DataGenerator


class KolmogorovGenerator(DataGenerator):
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_grid_size: Tuple[int, int, int],
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: int,
        max_num_trg: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.batch_grid_size = batch_grid_size
        self.min_num_ctx = torch.as_tensor(min_num_ctx, dtype=torch.int)
        self.max_num_ctx = torch.as_tensor(max_num_ctx, dtype=torch.int)
        self.min_num_trg = torch.as_tensor(min_num_trg, dtype=torch.int)
        self.max_num_trg = torch.as_tensor(max_num_trg, dtype=torch.int)

        fname = os.path.join(data_dir, split + ".h5")
        self.dataset, self.spatial_range, self.time_range = self.load_data(fname)

    def generate_batch(self) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Batch.
        """

        # Sample num_ctx and num_trg.
        nc, nt = self.sample_num_ctx_trg()

        # Sample dataset idx.
        dataset_idxs = self.sample_dataset_idx(self.batch_size)

        # Sample batch grids from within datasets.
        batch_grids = self.sample_batch_grid(self.batch_size)

        # Get the data.
        batch = self.sample_batch(
            num_ctx=nc, num_trg=nt, dataset_idxs=dataset_idxs, batch_grids=batch_grids
        )

        return batch

    def sample_dataset_idx(self, batch_size: int) -> List[int]:
        return random.sample(range(len(self.dataset)), batch_size)

    def sample_batch_grid(self, batch_size: int) -> List[Tuple[List, List, List]]:
        idx: List[Tuple[List, List, List]] = []
        start_spatial_idx1 = random.sample(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[0]), batch_size
        )
        start_spatial_idx2 = random.sample(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[1]), batch_size
        )
        start_time_idx = random.sample(
            range(len(self.time_range) - 1 - self.batch_grid_size[2]), batch_size
        )

        for i, j, k in zip(start_spatial_idx1, start_spatial_idx2, start_time_idx):
            idx.append(
                (
                    list(range(i, i + self.batch_grid_size[0])),
                    list(range(j, j + self.batch_grid_size[1])),
                    list(range(k, k + self.batch_grid_size[2])),
                )
            )

        return idx

    def sample_num_ctx_trg(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the numbers of context and target points in the dataset.

        Returns:
            num_ctx: Number of context points.
            num_trg: Number of target points.
        """

        # Sample number of context points
        num_ctx = torch.randint(
            low=self.min_num_ctx, high=self.max_num_ctx + 1, size=()
        )

        # Sample number of target points
        num_trg = torch.randint(
            low=self.min_num_trg, high=self.max_num_trg + 1, size=()
        )

        return num_ctx, num_trg

    def sample_batch(
        self,
        num_ctx: int,
        num_trg: int,
        dataset_idxs: List[int],
        batch_grids: List[Tuple[List, List, List]],
    ) -> Batch:
        x_list: List[torch.Tensor] = []
        x_shuffled_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        y_shuffled_list: List[torch.Tensor] = []
        for dataset_idx, batch_grid in zip(dataset_idxs, batch_grids):
            batch_grid_product = np.array(list(itertools.product(*batch_grid)))
            y = self.dataset[dataset_idx][
                batch_grid_product[:, 0],
                batch_grid_product[:, 1],
                batch_grid_product[:, 2],
            ]
            x = torch.stack(
                (
                    self.spatial_range[batch_grid_product[:, 0]],
                    self.spatial_range[batch_grid_product[:, 1]],
                    self.time_range[batch_grid_product[:, 2]],
                ),
                dim=-1,
            )

            # Now randomly select context and target points.
            shuffled_idx = list(range(len(x)))
            random.shuffle(shuffled_idx)

            x_list.append(x)
            y_list.append(y)
            x_shuffled_list.append(x[shuffled_idx])
            y_shuffled_list.append(y[shuffled_idx])

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        x_shuffled = torch.stack(x_shuffled_list)
        y_shuffled = torch.stack(y_shuffled_list)
        xc = x_shuffled[:, :num_ctx, :]
        yc = y_shuffled[:, :num_ctx, :]
        xt = x_shuffled[:, num_ctx : (num_ctx + num_trg), :]
        yt = y_shuffled[:, num_ctx : (num_ctx + num_trg), :]

        return Batch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
        )

    def load_data(self, fname: str, window: int = None) -> Tensor:
        with h5py.File(fname, mode="r") as f:
            data = f["x"][:]

        data = torch.as_tensor(data, dtype=torch.float)

        if window is None:
            pass
        elif window == 1:
            data = data.flatten(0, 1)
        else:
            data = data.unfold(1, window, 1)
            data = data.movedim(-1, 2)
            data = data.flatten(2, 3)
            data = data.flatten(0, 1)

        spatial_range = time_range = torch.linspace(-3, 3, 64)

        return data.permute(0, 1, 3, 4, 2), spatial_range, time_range
