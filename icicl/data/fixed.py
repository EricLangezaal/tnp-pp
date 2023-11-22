from typing import Optional

import torch

from .data import (
    ICSyntheticBatch,
    ICSyntheticGenerator,
    SyntheticBatch,
    SyntheticGenerator,
)


class FixedSizeSyntheticDataGenerator(SyntheticGenerator):
    def __init__(
        self,
        *,
        data_generator_class: type[SyntheticGenerator],
        num_datasets: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_datasets = num_datasets
        self.data_generator_class = data_generator_class

        # Generate fixed dataset.
        data_generator = data_generator_class(
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
        self.data = data_generator.generate_batch()

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


class FixedSizeICSyntheticDataGenerator(ICSyntheticGenerator):
    def __init__(
        self,
        *,
        data_generator_class: type[ICSyntheticGenerator],
        num_datasets: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_datasets = num_datasets

        # Generate fixed dataset.
        data_generator = data_generator_class(
            samples_per_epoch=num_datasets,
            batch_size=num_datasets,
            dim=self.dim,
            min_num_ctx=self.max_num_ctx,
            max_num_ctx=self.max_num_ctx,
            min_num_trg=self.max_num_trg,
            max_num_trg=self.max_num_trg,
            min_num_dc=self.max_num_dc,
            max_num_dc=self.max_num_dc,
            min_num_dc_ctx=self.max_num_dc_ctx,
            max_num_dc_ctx=self.max_num_dc_ctx,
            context_range=self.context_range,
            target_range=self.target_range,
            ic_context_range=self.ic_context_range,
        )
        self.data = data_generator.generate_batch()

    def sample_ic_batch(
        self, num_ctx: int, num_trg: int, num_dc: int, num_dc_ctx: int
    ) -> ICSyntheticBatch:
        # Sample a random subset of the fixed dataset.
        dataset_idx = torch.randperm(self.num_datasets)[: self.batch_size]

        # Sample random subset of dataset for context / target set.
        context_idx = torch.randperm(self.max_num_ctx)[:num_ctx]
        target_idx = torch.randperm(self.max_num_trg)[:num_trg]
        dc_idx = torch.randperm(self.max_num_dc)[:num_dc]
        dc_context_idx = torch.randperm(self.max_num_dc_ctx)[:num_dc_ctx]

        xc = self.data.xc[dataset_idx, ...][:, context_idx, :]
        yc = self.data.yc[dataset_idx, ...][:, context_idx, :]
        xt = self.data.xt[dataset_idx, ...][:, target_idx, :]
        yt = self.data.yt[dataset_idx, ...][:, target_idx, :]
        xic = self.data.xic[dataset_idx][:, dc_idx, ...][:, :, dc_context_idx, :]
        yic = self.data.yic[dataset_idx][:, dc_idx, ...][:, :, dc_context_idx, :]

        x = torch.cat([xc, xt], dim=1)
        y = torch.cat([yc, yt], dim=1)

        return ICSyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            xic=xic,
            yic=yic,
        )

    def sample_outputs(
        self,
        x: torch.Tensor,
        xic: Optional[torch.Tensor] = None,
    ):
        return None
