from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch


class GroundTruthPredictor(ABC):
    def __init__(self):
        pass

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Any:
        raise NotImplementedError


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor

    xc: torch.Tensor
    yc: torch.Tensor

    xt: torch.Tensor
    yt: torch.Tensor


@dataclass
class ICBatch(Batch):
    xic: torch.Tensor
    yic: torch.Tensor


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
        self.num_batches = samples_per_epoch // batch_size

        # Set batch counter.
        self.batch_counter = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """Reset batch counter and return self."""
        self.batch_counter = 0
        return self

    def __next__(self) -> Batch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batch_counter >= self.num_batches:
            raise StopIteration

        self.batch_counter += 1
        return self.generate_batch()

    @abstractmethod
    def generate_batch(self) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """
