from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import torch
import torchvision

from .base import Batch, ICBatch


@dataclass
class ImageBatch(Batch):
    label: torch.Tensor
    mc: torch.Tensor


@dataclass
class ICImageBatch(ImageBatch, ICBatch):
    mcdc: torch.Tensor
    xic_orig: torch.Tensor
    yic_orig: torch.Tensor


@dataclass
class GriddedImageBatch(ImageBatch):
    y_grid: torch.Tensor
    mc_grid: torch.Tensor
    mt_grid: torch.Tensor


class SingleLabelBatchSampler:
    def __init__(
        self, labels: torch.Tensor, batch_size: int, num_batches: Optional[int] = None
    ):
        # Get idxs for each label.
        self.label_list = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_list[label.item()].append(idx)

        # Create sampler for idxs of each label.
        self.samplers = {}
        for label, label_idxs in self.label_list.items():
            self.samplers[label] = torch.utils.data.RandomSampler(label_idxs)

        # Compute total number of batches.
        max_num_batches = min(
            len(sampler) // batch_size for sampler in self.samplers.values()
        ) * len(self.samplers)

        if num_batches is None:
            num_batches = max_num_batches

        self.batch_size = batch_size
        self.num_batches = min(max_num_batches, num_batches)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Reset samplers and batch label order.
        samplers = {label: iter(sampler) for label, sampler in self.samplers.items()}

        labels = torch.as_tensor(list(self.label_list.keys()))
        batch_labels = torch.cat(
            [
                labels[torch.randperm(len(labels))]
                for _ in range((self.num_batches // len(labels)) + 1)
            ],
            dim=0,
        )[: self.num_batches]

        for label in batch_labels:
            try:
                batch = [
                    self.label_list[int(label)][next(samplers[int(label)])]
                    for _ in range(self.batch_size)
                ]
                yield batch
            except StopIteration:
                yield None


class ImageGenerator:
    def __init__(
        self,
        *,
        dataset: torchvision.datasets.VisionDataset,
        dim: int,
        batch_size: int,
        min_prop_ctx: float,
        max_prop_ctx: float,
        samples_per_epoch: Optional[int] = None,
        same_label_per_batch: bool = False,
        x_mean: Optional[Tuple[float, float]] = None,
        x_std: Optional[Tuple[float, float]] = None,
        return_as_gridded: bool = False,
    ):
        self.batch_size = batch_size
        self.min_prop_ctx = min_prop_ctx
        self.max_prop_ctx = max_prop_ctx
        self.dim = dim
        self.dataset = dataset

        if samples_per_epoch is None:
            samples_per_epoch = len(self.dataset)

        self.samples_per_epoch = min(samples_per_epoch, len(self.dataset))
        self.num_batches = samples_per_epoch // batch_size
        self.same_label_per_batch = same_label_per_batch
        self.return_as_gridded = return_as_gridded

        # Create batch sampler.
        if self.same_label_per_batch:
            self.batch_sampler = iter(
                SingleLabelBatchSampler(
                    self.dataset.targets,
                    batch_size=batch_size,
                    num_batches=self.num_batches,
                )
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                self.dataset, num_samples=samples_per_epoch
            )
            self.batch_sampler = iter(
                torch.utils.data.BatchSampler(
                    sampler, batch_size=batch_size, drop_last=True
                )
            )

        # Set input mean and std.
        if x_mean is None or x_std is None:
            x = torch.stack(
                torch.meshgrid(
                    *[
                        torch.range(0, dim - 1)
                        for dim in self.dataset[0][0][0, ...].shape
                    ]
                ),
                dim=-1,
            )
            x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")
            self.x_mean = x.mean(dim=0)
            self.x_std = x.std(dim=0)
        else:
            self.x_mean = torch.as_tensor(x_mean)
            self.x_std = torch.as_tensor(x_std)

        # Set the batch counter.
        self.batches = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """Reset epoch counter and batch sampler and return self."""
        self.batches = 0
        # Create batch sampler.
        if self.same_label_per_batch:
            self.batch_sampler = iter(
                SingleLabelBatchSampler(
                    self.dataset.targets,
                    batch_size=self.batch_size,
                    num_batches=self.num_batches,
                )
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                self.dataset, num_samples=self.samples_per_epoch
            )
            self.batch_sampler = iter(
                torch.utils.data.BatchSampler(
                    sampler, batch_size=self.batch_size, drop_last=True
                )
            )
        return self

    def __next__(self) -> ImageBatch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batches >= self.num_batches:
            raise StopIteration

        self.batches += 1
        return self.generate_batch()

    def generate_batch(self) -> ImageBatch:
        """Generate batch of data.

        Returns:
            Batch: Tuple of tensors containing the context and target data.
        """

        # Sample context masks.
        pc = self.sample_prop(self.min_prop_ctx, self.max_prop_ctx)
        mc = self.sample_masks(prop=pc, batch_shape=torch.Size((self.batch_size,)))

        # Sample batch of data.
        batch = self.sample_batch(mc=mc)

        return batch

    def sample_prop(self, min_prop: float, max_prop: float) -> torch.Tensor:
        # Sample proportions to mask.
        prop = torch.rand(size=()) * (max_prop - min_prop) + min_prop

        return prop

    def sample_masks(self, prop: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """Sample context masks.

        Returns:
            mc: Context mask.
        """

        # Sample proportions to mask.
        num_mask = self.dim * prop
        rand = torch.rand(size=(*batch_shape, self.dim))
        randperm = rand.argsort(dim=-1)
        mc = randperm < num_mask

        return mc

    def sample_batch(self, mc: torch.Tensor) -> ImageBatch:
        """Sample batch of data.

        Args:
            mc: Context mask.

        Returns:
            batch: Batch of data.
        """

        # Sample batch of data.
        batch_idx = next(self.batch_sampler)
        y_grid = torch.cat([self.dataset[idx][0] for idx in batch_idx], dim=0)
        label = torch.stack(
            [torch.as_tensor(self.dataset[idx][1]) for idx in batch_idx]
        )

        # Input grid.
        x = torch.stack(
            torch.meshgrid(*[torch.range(0, dim - 1) for dim in y_grid[0, ...].shape]),
            dim=-1,
        )

        # Rearrange.
        y = einops.rearrange(y_grid, "m n1 n2 -> m (n1 n2) 1")
        x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")

        # Normalise inputs.
        x = (x - self.x_mean) / self.x_std
        x = einops.repeat(x, "n p -> m n p", m=len(batch_idx))

        xc = torch.stack([x_[mask] for x_, mask in zip(x, mc)])
        yc = torch.stack([y_[mask] for y_, mask in zip(y, mc)])
        xt = torch.stack([x_[~mask] for x_, mask in zip(x, mc)])
        yt = torch.stack([y_[~mask] for y_, mask in zip(y, mc)])

        if self.return_as_gridded:
            # Restructure mask.
            mc_grid = einops.rearrange(
                mc,
                "m (n1 n2) -> m n1 n2",
                n1=y_grid[0, ...].shape[0],
                n2=y_grid[0, ...].shape[1],
            )
            mt_grid = ~mc_grid
            return GriddedImageBatch(
                x=x,
                y=y,
                xc=xc,
                yc=yc,
                xt=xt,
                yt=yt,
                label=label,
                mc=mc,
                y_grid=y_grid,
                mc_grid=mc_grid,
                mt_grid=mt_grid,
            )

        return ImageBatch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt, label=label, mc=mc)


class ICImageGenerator(ImageGenerator):
    def __init__(
        self,
        *,
        min_num_dc: int,
        max_num_dc: int,
        min_prop_dc_ctx: float,
        max_prop_dc_ctx: float,
        batch_size: int,
        samples_per_epoch: int,
        **kwargs,
    ):
        self.actual_batch_size = batch_size

        # Set batch_size for sampling from SingleLabelBatchSampler
        samples_per_epoch *= 1 + max_num_dc
        batch_size = 1 + max_num_dc

        super().__init__(
            batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            same_label_per_batch=True,
            **kwargs,
        )

        self.min_num_dc = min_num_dc
        self.max_num_dc = max_num_dc
        self.min_prop_dc_ctx = min_prop_dc_ctx
        self.max_prop_dc_ctx = max_prop_dc_ctx

    def __len__(self):
        return self.num_batches // self.actual_batch_size

    def generate_batch(self) -> ICImageBatch:
        """Generate batch of data.

        Returns:
            Batch: Tuple of tensors containing the context and target data.
        """

        # Sample context masks.
        pc = self.sample_prop(self.min_prop_ctx, self.max_prop_ctx)
        mc = self.sample_masks(
            prop=pc, batch_shape=torch.Size((self.actual_batch_size,))
        )

        # Sample in-context datasets.
        num_dc = torch.randint(low=self.min_num_dc, high=self.max_num_dc + 1, size=())
        pcdc = self.sample_prop(self.min_prop_dc_ctx, self.max_prop_dc_ctx)
        mcdc = self.sample_masks(
            prop=pcdc, batch_shape=torch.Size((self.actual_batch_size, num_dc))
        )

        # Sample batch of data.
        batch = self.sample_ic_batch(mc=mc, mcdc=mcdc)

        return batch

    def sample_ic_batch(self, mc: torch.Tensor, mcdc: torch.Tensor) -> ICImageBatch:
        """Sample batch of data.

        Args:
            mc: Context mask.
            mcdc: Context masks for in-context datasets.

        Returns:
            batch: Batch of data.
        """

        batch_size = 1 + mcdc.shape[1]
        batch_idxs = []
        for _ in range(self.actual_batch_size):
            batch_idxs.append(next(self.batch_sampler)[: 1 + batch_size])

        # (m, 1 + ndc, n1, n2).
        y = torch.stack(
            [
                torch.cat([self.dataset[idx][0] for idx in batch_idx], dim=0)
                for batch_idx in batch_idxs
            ]
        )
        label = torch.stack(
            [
                torch.stack(
                    [torch.as_tensor(self.dataset[idx][1]) for idx in batch_idx]
                )
                for batch_idx in batch_idxs
            ]
        )

        # Input grid.
        x = torch.stack(
            torch.meshgrid(*[torch.range(0, dim - 1) for dim in y[0, 0, ...].shape]),
            dim=-1,
        )

        # Rearrange.
        y = einops.rearrange(y, "m np1 n1 n2 -> m np1 (n1 n2) 1")
        x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")

        # Normalise inputs.
        x = (x - self.x_mean) / self.x_std
        x = einops.repeat(
            x,
            "n p -> m np1 n p",
            m=self.actual_batch_size,
            np1=batch_size,
        )

        # Sample dataset.
        xc = torch.stack([x_[mask] for x_, mask in zip(x[:, 0, ...], mc)])
        yc = torch.stack([y_[mask] for y_, mask in zip(y[:, 0, ...], mc)])
        xt = torch.stack([x_[~mask] for x_, mask in zip(x[:, 0, ...], mc)])
        yt = torch.stack([y_[~mask] for y_, mask in zip(y[:, 0, ...], mc)])

        # Sample in-context dataset.
        xic = torch.stack(
            [
                torch.stack([x_[mask] for x_, mask in zip(x[i, 1:, ...], mcdc[i, ...])])
                for i in range(self.actual_batch_size)
            ]
        )
        yic = torch.stack(
            [
                torch.stack([y_[mask] for y_, mask in zip(y[i, 1:, ...], mcdc[i, ...])])
                for i in range(self.actual_batch_size)
            ]
        )

        return ICImageBatch(
            x=x[:, 0, ...],
            y=y[:, 0, ...],
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            label=label,
            mc=mc,
            xic=xic,
            yic=yic,
            mcdc=mcdc,
            xic_orig=x[:, 1:, ...],
            yic_orig=y[:, 1:, ...],
        )
