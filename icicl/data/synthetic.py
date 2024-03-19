import operator
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import Batch, DataGenerator, GroundTruthPredictor


@dataclass
class SyntheticBatch(Batch):
    gt_mean: Optional[torch.Tensor] = None
    gt_std: Optional[torch.Tensor] = None

    gt_loglik: Optional[torch.Tensor] = None
    gt_pred: Optional[GroundTruthPredictor] = None

    gt_lengthscale: Optional[torch.Tensor] = None


class SyntheticGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        dim: int,
        min_num_ctx: int,
        max_num_ctx: int,
        min_num_trg: int,
        max_num_trg: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set synthetic generator parameters
        self.dim = dim
        self.min_num_ctx = torch.as_tensor(min_num_ctx, dtype=torch.int)
        self.max_num_ctx = torch.as_tensor(max_num_ctx, dtype=torch.int)
        self.min_num_trg = torch.as_tensor(min_num_trg, dtype=torch.int)
        self.max_num_trg = torch.as_tensor(max_num_trg, dtype=torch.int)

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """

        if batch_shape is None:
            batch_shape = torch.Size((self.batch_size,))

        # Sample number of context and target points
        num_ctx, num_trg = self.sample_num_ctx_trg()

        # Sample entire batch (context and target points)
        batch = self.sample_batch(
            num_ctx=num_ctx,
            num_trg=num_trg,
            batch_shape=batch_shape,
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

    def sample_batch(
        self,
        num_ctx: int,
        num_trg: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        # Sample inputs, then outputs given inputs
        x = self.sample_inputs(
            num_ctx=num_ctx, num_trg=num_trg, batch_shape=batch_shape
        )
        y, gt_pred = self.sample_outputs(x=x)

        xc = x[:, :num_ctx, :]
        yc = y[:, :num_ctx, :]
        xt = x[:, num_ctx:, :]
        yt = y[:, num_ctx:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred,
        )

    @abstractmethod
    def sample_inputs(
        self,
        num_ctx: int,
        batch_shape: torch.Size,
        num_trg: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        """Sample context and target outputs, given the inputs `x`.

        Arguments:
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.

        Returns:
            y: Tensor of shape (batch_size, num_ctx + num_trg, 1) containing
                the context and target outputs.
        """


class SyntheticGeneratorUniformInput(SyntheticGenerator):
    def __init__(
        self,
        *,
        context_range: Tuple[float, float],
        target_range: Tuple[float, float],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_range = torch.as_tensor(context_range, dtype=torch.float)
        self.target_range = torch.as_tensor(target_range, dtype=torch.float)

    def sample_inputs(
        self,
        num_ctx: int,
        batch_shape: torch.Size,
        num_trg: Optional[int] = None,
    ) -> torch.Tensor:
        xc = (
            torch.rand((*batch_shape, num_ctx, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            + self.context_range[:, 0]
        )

        if num_trg is not None:
            xt = (
                torch.rand((*batch_shape, num_trg, self.dim))
                * (self.target_range[:, 1] - self.target_range[:, 0])
                + self.target_range[:, 0]
            )

            return torch.concat([xc, xt], axis=1)

        return xc


class SyntheticGeneratorBimodalInput(SyntheticGenerator):
    def __init__(
        self,
        *,
        context_range: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...],
        target_range: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...],
        mode_offset_range: Tuple[Tuple[float, float], ...] = ((0.0, 0.0),),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_range = torch.as_tensor(context_range, dtype=torch.float)
        self.target_range = torch.as_tensor(target_range, dtype=torch.float)
        self.mode_offset_range = torch.as_tensor(mode_offset_range, dtype=torch.float)

    def sample_inputs(
        self,
        num_ctx: int,
        batch_shape: torch.Size,
        num_trg: Optional[int] = None,
    ) -> torch.Tensor:

        # Sample the mode.
        ctx_bernoulli_probs = torch.empty(*batch_shape, num_ctx).fill_(0.5)
        ctx_modes = torch.bernoulli(ctx_bernoulli_probs).int()

        # Apply offset to the range of each mode.
        mode_offset = (
            torch.rand((self.dim, 2))
            * (self.mode_offset_range[..., 1] - self.mode_offset_range[..., 0])
            + self.mode_offset_range[..., 0]
        )
        context_range = self.context_range + mode_offset[..., None]
        target_range = self.target_range + mode_offset[..., None]

        xc = torch.rand((*batch_shape, num_ctx, self.dim)) * (
            context_range[..., ctx_modes, 1].permute(1, 2, 0)
            - context_range[..., ctx_modes, 0].permute(1, 2, 0)
        ) + context_range[..., ctx_modes, 0].permute(1, 2, 0)

        if num_trg is not None:
            # Sample the mode.
            trg_bernoulli_probs = torch.empty(*batch_shape, num_trg).fill_(0.5)
            trg_modes = torch.bernoulli(trg_bernoulli_probs).int()
            xt = torch.rand((*batch_shape, num_trg, self.dim)) * (
                target_range[..., trg_modes, 1].permute(1, 2, 0)
                - target_range[..., trg_modes, 0].permute(1, 2, 0)
            ) + target_range[..., trg_modes, 0].permute(1, 2, 0)

            return torch.concat([xc, xt], axis=1)

        return xc


class SyntheticGeneratorMixture(SyntheticGenerator):
    def __init__(
        self,
        *,
        generators: Tuple[SyntheticGenerator, ...],
        mixture_probs: Tuple[float, ...],
        mix_samples: bool = False,
        **kwargs,
    ):
        assert len(generators) == len(
            mixture_probs
        ), "Must be a mixture prob for each generator."
        assert sum(mixture_probs) == 1, "Sum of mixture_probs must be 1."
        assert all(
            prob > 0 for prob in mixture_probs
        ), "All elements of mixture_probs must be positive."

        super().__init__(**kwargs)

        # Whether or not to sample mixture for each sample in batch.
        self.mix_samples = mix_samples
        self.generators = generators
        self.mixture_probs = mixture_probs

        # Ensure samples per epoch of generators are infinite, so does not stop sampling.
        for generator in self.generators:
            generator.num_batches = np.inf

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        if self.mix_samples:
            # First, sample which generator the batch is sampled from.
            gens = random.choices(
                self.generators,
                weights=self.mixture_probs,
                k=self.batch_size,
            )
            gen_counter = Counter(gens)

            # Sample nc and nt which is shared for all samples.
            nc, nt = self.sample_num_ctx_trg()

            sub_batches: List[Batch] = []
            for gen, batch_size in gen_counter.items():
                batch_shape = torch.Size((batch_size,))
                sub_batch = gen.sample_batch(
                    num_ctx=nc, num_trg=nt, batch_shape=batch_shape
                )
                sub_batches.append(sub_batch)

            random.shuffle(sub_batches)

            # Combine sub_batches.
            # batch = sum(sub_batches)
            batch = reduce(operator.add, sub_batches)
            return batch

        gen = random.choices(self.generators, weights=self.mixture_probs, k=1)[0]

        # Sample nc and nt which is shared for all samples.
        nc, nt = self.sample_num_ctx_trg()
        return gen.sample_batch(
            num_ctx=nc, num_trg=nt, batch_shape=torch.Size((self.batch_size,))
        )

    def sample_inputs(
        self,
        num_ctx: int,
        batch_shape: torch.Size,
        num_trg: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        raise NotImplementedError
