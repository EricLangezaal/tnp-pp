from typing import Optional

import torch

from .data import SyntheticBatch, SyntheticGenerator


class FixedSizeSyntheticDataGenerator(SyntheticGenerator):
    def __init__(
        self,
        *,
        data_generator: type[SyntheticGenerator],
        num_datasets: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_datasets = num_datasets

        # Generate fixed dataset.
        data_generator_ = data_generator(
            samples_per_epoch=num_datasets,
            batch_size=num_datasets,
            dim=self.dim,
            min_num_ctx=self.max_num_ctx,
            max_num_ctx=self.max_num_ctx,
            min_num_trg=self.max_num_trg,
            max_num_trg=self.max_num_trg,
            context_range=self.context_range,
            target_range=self.target_range,
        )
        self.data = data_generator_.generate_batch()

    def sample_batch(self, num_ctx: int, num_trg: int) -> SyntheticBatch:
        # Sample a random subset of the fixed dataset.
        dataset_idx = torch.randperm(self.num_datasets)[: self.batch_size]

        # Sample random subset of dataset for context / target set.
        context_idx = torch.randperm(self.max_num_ctx)[:num_ctx]
        target_idx = torch.randperm(self.max_num_trg)[:num_trg]

        xc = self.data.xc[dataset_idx, ...][:, context_idx, :]
        yc = self.data.yc[dataset_idx, ...][:, context_idx, :]
        xt = self.data.xt[dataset_idx, ...][:, target_idx, :]
        yt = self.data.yt[dataset_idx, ...][:, target_idx, :]

        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
        xic: Optional[torch.Tensor] = None,
    ):
        return None
