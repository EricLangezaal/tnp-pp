from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


class GroundTruthPredictor(ABC):
    def __init__(self):
        pass

    def __call__(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor
    ) -> Any:
        raise NotImplementedError


@dataclass
class Batch:
    x: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None

    xc: Optional[torch.Tensor] = None
    yc: Optional[torch.Tensor] = None
    xt: Optional[torch.Tensor] = None
    yt: Optional[torch.Tensor] = None


@dataclass
class SyntheticBatch(Batch):
    gt_mean: Optional[torch.Tensor] = None
    gt_std: Optional[torch.Tensor] = None

    gt_loglik: Optional[torch.Tensor] = None
    gt_pred: Optional[GroundTruthPredictor] = None

    gt_lengthscale: Optional[torch.Tensor] = None


class DataGenerator(ABC):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
    ):
        """Base data generator, which can be used to derive other data generators,
        such as synthetic generators or real data generators.

        Arguments:
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
        """

        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.num_batches = samples_per_epoch // batch_size + 1

        # Set epoch counter.
        self.epoch = 0

    def __iter__(self):
        """Reset epoch counter and return self."""
        self.epoch = 0
        return self

    def __next__(self) -> Batch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.epoch >= self.num_batches:
            raise StopIteration

        self.epoch += 1
        return self.generate_batch()

    @abstractmethod
    def generate_batch(self) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """


class SyntheticGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        dim: int,
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: int,
        max_num_trg: int,
        context_range: torch.Tensor,
        target_range: torch.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set synthetic generator parameters
        self.dim = dim
        self.min_num_ctx = torch.as_tensor(min_num_ctx, dtype=torch.int)
        self.max_num_ctx = torch.as_tensor(max_num_ctx, dtype=torch.int)
        self.min_num_trg = torch.as_tensor(min_num_trg, dtype=torch.int)
        self.max_num_trg = torch.as_tensor(max_num_trg, dtype=torch.int)

        self.context_range = torch.as_tensor(context_range, dtype=torch.float)
        self.target_range = torch.as_tensor(target_range, dtype=torch.float)

    def generate_batch(self) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """

        # Sample number of context and target points
        num_ctx, num_trg = self.sample_num_ctx_trg()

        # Sample entire batch (context and target points)
        batch = self.sample_batch(
            num_ctx=num_ctx,
            num_trg=num_trg,
        )

        return batch

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

    def sample_batch(self, num_ctx: int, num_trg: int) -> Batch:
        # Sample inputs, then outputs given inputs
        x = self.sample_inputs(num_ctx=num_ctx, num_trg=num_trg)
        y, gt_pred = self.sample_outputs(x=x)

        xc = x[:, :num_ctx, :]
        yc = y[:, :num_ctx, :]
        xt = x[:, num_ctx:, :]
        yt = y[:, num_ctx:, :]

        return SyntheticBatch(
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred,
        )

    def sample_inputs(self, num_ctx: int, num_trg: int) -> torch.Tensor:
        """Sample context and target inputs, sampled uniformly from the boxes
        defined by `context_range` and `target_range` respectively.

        Arguments:
            seed: Random seed.
            num_ctx: Number of context points.
            num_trg: Number of target points.

        Returns:
            seed: Random seed generated by splitting.
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.
        """

        xc = (
            torch.rand((self.batch_size, num_ctx, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            - self.context_range[:, 0]
        )

        xt = (
            torch.rand((self.batch_size, num_trg, self.dim))
            * (self.target_range[:, 1] - self.target_range[:, 0])
            - self.target_range[:, 0]
        )

        return torch.concat([xc, xt], axis=1)

    @abstractmethod
    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        """Sample context and target outputs, given the inputs `x`.

        Arguments:
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.

        Returns:
            y: Tensor of shape (batch_size, num_ctx + num_trg, 1) containing
                the context and target outputs.
        """
