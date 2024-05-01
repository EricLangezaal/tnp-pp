from dataclasses import dataclass
from typing import Tuple, Optional
import torch

from .base import Batch, DataGenerator
from .synthetic import SyntheticGenerator
from ..utils.conv import make_grid, flatten_grid

@dataclass
class OOTGBatch:
    xc_on_grid: torch.Tensor
    yc_on_grid: torch.Tensor

    xc_off_grid: torch.Tensor
    yc_off_grid: torch.Tensor

    xt: torch.Tensor
    yt: torch.Tensor


class SyntheticOOTGGenerator(DataGenerator):
    def __init__(
        self,
        *,
        off_grid_generator: SyntheticGenerator,
        grid_range: Tuple[Tuple[float, float], ...], # so pair for each dimension
        points_per_unit: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.otg_generator = off_grid_generator
        self.grid_range = torch.as_tensor(grid_range, dtype=torch.float)
        self.points_per_unit = torch.as_tensor(points_per_unit)

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
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
        ontg_x = make_grid(
            xmin = self.grid_range[:, 0].repeat(*batch_shape, 1), 
            xmax = self.grid_range[:, 1].repeat(*batch_shape, 1), 
            points_per_unit = self.points_per_unit, 
            margin = 0)
        ontg_x = flatten_grid(ontg_x) # shape (batch, num_ontg, xdim)

        # (batch_shape, num_ctx + num_trg + num_ontg, xdim).
        x = torch.cat((offtg_x, ontg_x), dim=-2)

        # (batch_shape, num_ctx + num_trg + num_ontg, 1).
        y, _  = self.otg_generator.sample_outputs(x=x)
        offtg_y = y[:, :offtg_x.shape[-2], :]
        ontg_y = y[:, offtg_x.shape[-2]:, :]

        offtg_xc = offtg_x[:, :num_ctx, :]
        offtg_yc = offtg_y[:, :num_ctx, :]

        xt = offtg_x[:, num_ctx:, :]
        yt = offtg_y[:, num_ctx:, :]

        return OOTGBatch(
            xc_off_grid=offtg_xc,
            yc_off_grid=offtg_yc,
            xc_on_grid=ontg_x,
            yc_on_grid=ontg_y,
            xt=xt,
            yt=yt,
        )