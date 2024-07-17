from dataclasses import dataclass
from typing import Tuple, Optional, Union
from typing_extensions import Self
from math import prod
from enum import Enum, auto

import torch
import warnings

from .base import DataGenerator
from .synthetic import SyntheticGenerator, SyntheticBatch
from ..utils.grids import unflatten_grid, flatten_grid, make_grid_from_range

class DataModality(Enum):
    ON_GRID = auto()
    OFF_GRID = auto()
    BOTH = auto()

    def parse(value: Union[str, Self]) -> Self:
        if isinstance(value, DataModality):
            return value
        try:
            return DataModality[str(value).upper()]
        except KeyError:
            warnings.warn(f"Invalid value for DataModality: {value}. Defaulting to BOTH!")
            return DataModality.BOTH

    def get(
            self, 
            on_grid: torch.Tensor, 
            off_grid: torch.Tensor, 
            both: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self == DataModality.ON_GRID:
            return on_grid
        elif self == DataModality.OFF_GRID:
            return off_grid
        elif isinstance(both, torch.Tensor):
            return both
        return torch.cat((off_grid, flatten_grid(on_grid)), dim=-2)

@dataclass
class OOTGBatch(SyntheticBatch):
    xc_on_grid: torch.Tensor
    yc_on_grid: torch.Tensor

    xc_off_grid: torch.Tensor
    yc_off_grid: torch.Tensor

    used_modality: DataModality = DataModality.BOTH


class SyntheticOOTGGenerator(DataGenerator):
    def __init__(
        self,
        *,
        off_grid_generator: SyntheticGenerator,
        grid_range: Tuple[Tuple[float, float], ...], # so pair for each dimension
        points_per_unit: int,
        used_modality: DataModality = DataModality.BOTH,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.otg_generator = off_grid_generator
        self.grid_range = torch.as_tensor(grid_range, dtype=torch.float)
        self.points_per_unit = torch.as_tensor(points_per_unit)
        self.used_modality = DataModality.parse(used_modality)

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> OOTGBatch:
        """
        Generate batch of data.
        Returns:
        batch: Tuple of tensors containing the context and target data.
        """

        if batch_shape is None:
            batch_shape = torch.Size((self.batch_size,))

        # Sample number of context and target points
        num_ctx, num_trg = self.otg_generator.sample_num_ctx_trg()

        # Sample entire batch (context and target points)
        batch = self.sample_batch(
            num_ctx=num_ctx,
            num_trg=num_trg,
            batch_shape=batch_shape,
        )
        return batch

    def sample_batch(self, 
        num_ctx: int, 
        num_trg: int, 
        batch_shape: torch.Size
    ) -> OOTGBatch:
        offtg_x = self.otg_generator.sample_inputs(
            num_ctx=num_ctx,
            num_trg=num_trg,
            batch_shape=batch_shape
        )
        ontg_x = make_grid_from_range(self.grid_range, self.points_per_unit, batch_shape=batch_shape)
        grid_shape = ontg_x.shape[1:-1]
        ontg_x = flatten_grid(ontg_x) # shape (batch, num_ontg, xdim)

        # (batch_shape, num_ctx + num_trg + num_ontg, xdim).
        x = torch.cat((offtg_x, ontg_x), dim=-2)

        # (batch_shape, num_ctx + num_trg + num_ontg, 1).
        # can be sampled from two Hadamard correlated processes though!
        y, gt_pred = self.otg_generator.sample_outputs(x=x, num_offtg=offtg_x.shape[-2])
        offtg_y = y[:, :offtg_x.shape[-2], :]
        ontg_y = y[:, offtg_x.shape[-2]:, :]

        offtg_xc = offtg_x[:, :num_ctx, :]
        offtg_yc = offtg_y[:, :num_ctx, :]

        xc = torch.cat((offtg_xc, ontg_x), dim=-2)
        yc = torch.cat((offtg_yc, ontg_y), dim=-2)

        xt = offtg_x[:, num_ctx:, :]
        yt = offtg_y[:, num_ctx:, :]

        return OOTGBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xc_off_grid=offtg_xc,
            yc_off_grid=offtg_yc,
            xc_on_grid=unflatten_grid(ontg_x, grid_shape=grid_shape),
            yc_on_grid=unflatten_grid(ontg_y, grid_shape=grid_shape),
            xt=xt,
            yt=yt,
            gt_pred=gt_pred,
            used_modality=self.used_modality,
        )

    
class RandomOOTGGenerator(DataGenerator):

    def __init__(self, *, num_off_grid_context: int, grid_shape: Tuple[int, ...], num_targets: int, dim: int =1, **kwargs):
        super().__init__(**kwargs)
        self.num_off_grid_context = num_off_grid_context
        self.grid_shape = tuple(grid_shape)
        self.num_targets = num_targets
        self.dim = dim

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> OOTGBatch:
        if batch_shape is None:
            batch_shape = torch.Size((self.batch_size,))

        batch = self.sample_batch(num_ctx=self.num_off_grid_context, num_trg=self.num_targets, batch_shape=batch_shape)
        return batch

    def sample_batch(self, num_ctx: int, num_trg: int, batch_shape: torch.Size) -> OOTGBatch:

        x_off_grid = torch.randn(*batch_shape, num_ctx + num_trg, self.dim)
        y_off_grid = torch.randn(x_off_grid.shape[:-1] + (1,))

        x_on_grid = torch.randn(*batch_shape, prod(self.grid_shape), self.dim)
        y_on_grid = torch.randn(x_on_grid.shape[:-1] + (1,))

        return OOTGBatch(
            x=torch.cat((x_off_grid, x_on_grid), dim=-2),
            y=torch.cat((y_off_grid, y_on_grid), dim=-2),
            xc=torch.cat((x_off_grid[:, :num_ctx, :], x_on_grid), dim=-2),
            yc=torch.cat((y_off_grid[:, :num_ctx, :], y_on_grid), dim=-2),
            xc_off_grid=x_off_grid[:, :num_ctx, :],
            yc_off_grid=y_off_grid[:, :num_ctx, :],
            xc_on_grid=unflatten_grid(x_on_grid, self.grid_shape),
            yc_on_grid=unflatten_grid(y_on_grid, self.grid_shape),
            xt=x_off_grid[:, num_ctx:, :],
            yt=y_off_grid[:, num_ctx:, :],
            gt_pred=None,
        )
