from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torchvision

from .data import Batch


@dataclass
class MNISTBatch(Batch):
    label: Optional[torch.Tensor] = None
    mc: Optional[torch.Tensor] = None


class SingleLabelBatchSampler:
    def __init__(
        self, labels: torch.Tensor, batch_size: int, num_batches: Optional[int] = None
    ):
        self.labels = labels
        self.unique_labels = torch.unique(labels)
        self.batch_size = batch_size

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

        self.num_batches = min(max_num_batches, num_batches)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Reset samplers and batch label order.
        samplers = {label: iter(sampler) for label, sampler in self.samplers.items()}

        batch_labels = torch.cat(
            [
                self.unique_labels[torch.randperm(len(self.unique_labels))]
                for _ in range((self.num_batches // len(self.unique_labels)) + 1)
            ],
            dim=0,
        )[: self.num_batches]

        for label in batch_labels:
            batch = [
                self.label_list[int(label)][next(samplers[int(label)])]
                for _ in range(self.batch_size)
            ]
            yield batch


class MNISTGenerator:
    def __init__(
        self,
        *,
        batch_size: int,
        min_prop_ctx: float,
        max_prop_ctx: float,
        data_dir: str,
        train: bool = True,
        download: bool = False,
        samples_per_epoch: Optional[int] = None,
        same_label_per_batch: bool = False,
    ):
        self.batch_size = batch_size
        self.min_prop_ctx = min_prop_ctx
        self.max_prop_ctx = max_prop_ctx
        self.dim = 28 * 28

        # Get MNIST dataset.
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        if samples_per_epoch is None:
            samples_per_epoch = len(self.dataset)

        self.samples_per_epoch = min(samples_per_epoch, len(self.dataset))
        self.num_batches = samples_per_epoch // batch_size
        self.same_label_per_batch = same_label_per_batch

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

        # Set the batch counter.
        self.batches = 0

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

    def __next__(self) -> Batch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batches >= self.num_batches:
            raise StopIteration

        self.batches += 1
        return self.generate_batch()

    def generate_batch(self) -> MNISTBatch:
        """Generate batch of data.

        Returns:
            Batch: Tuple of tensors containing the context and target data.
        """

        # Sample context masks.
        mc = self.sample_masks()

        # Sample batch of data.
        batch = self.sample_batch(mc=mc)

        return batch

    def sample_masks(self) -> torch.Tensor:
        """Sample context masks.

        Returns:
            mc: Context mask.
        """

        # Sample proportions to mask.
        prop_ctx = (
            torch.rand(size=()) * (self.max_prop_ctx - self.min_prop_ctx)
            + self.min_prop_ctx
        )
        num_mask = self.dim * prop_ctx
        mc = torch.stack(
            [torch.randperm(self.dim) < num_mask for _ in range(self.batch_size)]
        )

        return mc

    def sample_batch(self, mc: torch.Tensor) -> MNISTBatch:
        """Sample batch of data.

        Args:
            mc: Context mask.

        Returns:
            batch: Batch of data.
        """

        # Sample batch of data.
        batch_idx = next(self.batch_sampler)
        y = torch.cat([self.dataset[idx][0] for idx in batch_idx], dim=0)
        label = torch.stack(
            [torch.as_tensor(self.dataset[idx][1]) for idx in batch_idx]
        )

        # Input grid.
        x = torch.stack(
            torch.meshgrid(*[torch.range(0, dim - 1) for dim in y[0, ...].shape]),
            dim=-1,
        )

        # Rearrange.
        y = einops.rearrange(y, "m n1 n2 -> m (n1 n2) 1")
        x = einops.rearrange(x, "n1 n2 d -> (n1 n2) d")
        x = torch.reshape(x, shape=(-1, x.shape[-1]))

        # Normalise inputs.
        x = (x - x.mean(dim=-2)) / x.std(dim=-2)
        x = einops.repeat(x, "n p -> m n p", m=len(batch_idx))

        xc = torch.stack([x_[mask] for x_, mask in zip(x, mc)])
        yc = torch.stack([y_[mask] for y_, mask in zip(y, mc)])
        xt = torch.stack([x_[~mask] for x_, mask in zip(x, mc)])
        yt = torch.stack([y_[~mask] for y_, mask in zip(y, mc)])

        return MNISTBatch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt, label=label, mc=mc)
