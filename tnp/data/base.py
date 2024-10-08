from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from tqdm import tqdm


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

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor

    xc: torch.Tensor
    yc: torch.Tensor

    xt: torch.Tensor
    yt: torch.Tensor

    def __add__(self, b):
        return Batch(
            x=torch.cat((self.x, b.x), dim=0),
            y=torch.cat((self.y, b.y), dim=0),
            xc=torch.cat((self.xc, b.xc), dim=0),
            yc=torch.cat((self.yc, b.yc), dim=0),
            xt=torch.cat((self.xt, b.xt), dim=0),
            yt=torch.cat((self.yt, b.yt), dim=0),
        )

    def __radd__(self, b):
        return self.__add__(b)


class DataGenerator(torch.utils.data.IterableDataset, ABC):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        deterministic: bool = False,
        deterministic_seed: int = 0,
        num_workers: int = 0,
        **kwargs,
    ):
        """Base data generator, which can be used to derive other data generators,
        such as synthetic generators or real data generators.

        Arguments:
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
        """
        super().__init__(**kwargs)

        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.num_batches = samples_per_epoch // batch_size
        self.num_workers = num_workers

        # Set batch counter.
        self.batch_counter = 0
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed
        self.batches = None

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """Reset batch counter and return self."""
        if self.deterministic and self.batches is None:
            # Set deterministic seed.
            current_seed = torch.seed()
            torch.manual_seed(self.deterministic_seed)
            self.batches = [self.generate_batch() for _ in tqdm(range(self.num_batches), desc="Batch generation")]
            torch.manual_seed(current_seed)

        self.batch_counter = 0
        return self

    def __next__(self) -> Batch:
        """Generate next batch of data, using the `generate_batch` method.
        The `generate_batch` method should be implemented by the derived class.
        """

        if self.batch_counter >= self.num_batches:
            raise StopIteration

        if self.deterministic and self.batches is not None:
            batch = self.batches[self.batch_counter]
        else:
            batch = self.generate_batch()

        self.batch_counter += 1
        return batch

    @abstractmethod
    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """
