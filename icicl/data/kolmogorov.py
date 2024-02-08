import itertools
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from .base import Batch, DataGenerator


@dataclass
class KolmogorovBatch(Batch):
    re: List[int]


class KolmogorovGenerator(DataGenerator):
    def __init__(
        self,
        data_dir: str,
        re_dirs: Dict[int, str],
        split: str,
        batch_grid_size: Tuple[int, int, int],
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: int,
        max_num_trg: int,
        forecast_window: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.batch_grid_size = batch_grid_size
        self.min_num_ctx = torch.as_tensor(min_num_ctx, dtype=torch.int)
        self.max_num_ctx = torch.as_tensor(max_num_ctx, dtype=torch.int)
        self.min_num_trg = torch.as_tensor(min_num_trg, dtype=torch.int)
        self.max_num_trg = torch.as_tensor(max_num_trg, dtype=torch.int)
        self.forecast_window = forecast_window

        self.dataset, self.spatial_range, self.time_range = self.load_data(
            data_dir, re_dirs, split
        )

    def generate_batch(
        self, batch_shape: Optional[torch.Size] = None
    ) -> KolmogorovBatch:
        """Generate batch of data.

        Returns:
            batch: Batch.
        """
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]

        # Sample num_ctx and num_trg.
        nc, nt = self.sample_num_ctx_trg()

        # Sample reynold numbers.
        re_numbers = self.sample_reynold_numbers(batch_size)

        # Sample dataset idx.
        dataset_idxs = self.sample_dataset_idx(re_numbers, batch_size)

        # Sample batch grids from within datasets.
        batch_grids = self.sample_batch_grid(batch_size)

        # Get the data.
        batch = self.sample_batch(
            num_ctx=nc,
            num_trg=nt,
            re_numbers=re_numbers,
            dataset_idxs=dataset_idxs,
            batch_grids=batch_grids,
        )

        return batch

    def sample_reynold_numbers(self, batch_size: int) -> List[int]:
        return random.choices(list(self.dataset.keys()), k=batch_size)

    def sample_dataset_idx(self, re_numbers: List[int], batch_size: int) -> List[int]:
        assert len(re_numbers) == batch_size
        return [random.choice(range(len(self.dataset[re]))) for re in re_numbers]

    def sample_batch_grid(self, batch_size: int) -> List[Tuple[List, List, List]]:
        idx: List[Tuple[List, List, List]] = []
        start_spatial_idx1 = random.sample(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[0]), batch_size
        )
        start_spatial_idx2 = random.sample(
            range(len(self.spatial_range) - 1 - self.batch_grid_size[1]), batch_size
        )

        if self.forecast_window is None:
            start_time_idx = random.sample(
                range(len(self.time_range) - 1 - self.batch_grid_size[2]), batch_size
            )
        else:
            start_time_idx = random.sample(
                range(
                    len(self.time_range)
                    - 1
                    - self.batch_grid_size[2]
                    - self.forecast_window
                ),
                batch_size,
            )

        for i, j, k in zip(start_spatial_idx1, start_spatial_idx2, start_time_idx):
            if self.forecast_window is None:
                idx.append(
                    (
                        list(range(i, i + self.batch_grid_size[0])),
                        list(range(j, j + self.batch_grid_size[1])),
                        list(range(k, k + self.batch_grid_size[2])),
                    )
                )
            else:
                idx.append(
                    (
                        list(range(i, i + self.batch_grid_size[0])),
                        list(range(j, j + self.batch_grid_size[1])),
                        list(
                            range(k, k + self.batch_grid_size[2] + self.forecast_window)
                        ),
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
        re_numbers: List[int],
        dataset_idxs: List[int],
        batch_grids: List[Tuple[List, List, List]],
    ) -> KolmogorovBatch:
        x_list, xc_list, xt_list = [], [], []
        y_list, yc_list, yt_list = [], [], []
        re_list = []
        for re, dataset_idx, batch_grid in zip(re_numbers, dataset_idxs, batch_grids):
            re_list.append(re)
            if self.forecast_window is None:
                batch_grid_product = np.array(list(itertools.product(*batch_grid)))
                y = self.dataset[re][dataset_idx][
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
                xc_list.append(x[shuffled_idx[:num_ctx]])
                yc_list.append(y[shuffled_idx[:num_ctx]])
                xt_list.append(x[shuffled_idx[num_ctx : (num_ctx + num_trg)]])
                yt_list.append(y[shuffled_idx[num_ctx : (num_ctx + num_trg)]])

            else:
                context_batch_grid_product = np.asarray(
                    list(
                        itertools.product(
                            batch_grid[0],
                            batch_grid[1],
                            batch_grid[2][: -self.forecast_window],
                        )
                    )
                )
                target_batch_grid_product = np.asarray(
                    list(
                        itertools.product(
                            batch_grid[0],
                            batch_grid[1],
                            batch_grid[2][-self.forecast_window :],
                        )
                    )
                )

                yc_all = self.dataset[re][dataset_idx][
                    context_batch_grid_product[:, 0],
                    context_batch_grid_product[:, 1],
                    context_batch_grid_product[:, 2],
                ]
                xc_all = torch.stack(
                    (
                        self.spatial_range[context_batch_grid_product[:, 0]],
                        self.spatial_range[context_batch_grid_product[:, 1]],
                        self.time_range[context_batch_grid_product[:, 2]],
                    ),
                    dim=-1,
                )

                yt_all = self.dataset[re][dataset_idx][
                    target_batch_grid_product[:, 0],
                    target_batch_grid_product[:, 1],
                    target_batch_grid_product[:, 2],
                ]
                xt_all = torch.stack(
                    (
                        self.spatial_range[target_batch_grid_product[:, 0]],
                        self.spatial_range[target_batch_grid_product[:, 1]],
                        self.time_range[target_batch_grid_product[:, 2]],
                    ),
                    dim=-1,
                )

                # Now randomly select context and target points.
                context_shuffled_idx = list(range(len(xc_all)))
                random.shuffle(context_shuffled_idx)
                target_shuffled_idx = list(range(len(xt_all)))
                random.shuffle(target_shuffled_idx)

                x_list.append(xt_all)
                y_list.append(yt_all)
                xc_list.append(xc_all[context_shuffled_idx[:num_ctx]])
                yc_list.append(yc_all[context_shuffled_idx[:num_ctx]])
                xt_list.append(xt_all[target_shuffled_idx[:num_trg]])
                yt_list.append(yt_all[target_shuffled_idx[:num_trg]])

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        xc = torch.stack(xc_list)
        yc = torch.stack(yc_list)
        xt = torch.stack(xt_list)
        yt = torch.stack(yt_list)

        return KolmogorovBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            re=re_list,
        )

    def load_data(
        self, data_dir: str, re_dirs: Dict[int, str], split: str = "train"
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        data: Dict[int, torch.Tensor] = {}
        for re, re_dir in re_dirs.items():
            fname = os.path.join(data_dir, re_dir, split + ".h5")
            with h5py.File(fname, mode="r") as f:
                re_data = f["x"][:]

            re_data = torch.as_tensor(re_data, dtype=torch.float)
            data[re] = re_data.permute(0, 1, 3, 4, 2)

        spatial_range = time_range = torch.linspace(-3, 3, 64)

        return data, spatial_range, time_range
