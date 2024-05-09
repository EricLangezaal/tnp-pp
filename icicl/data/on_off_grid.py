from dataclasses import dataclass
from typing import Tuple, Optional
import torch

from .base import DataGenerator
from .synthetic import SyntheticGenerator, SyntheticBatch
from ..utils.conv import make_grid, flatten_grid

@dataclass
class OOTGBatch(SyntheticBatch):
    xc_on_grid: torch.Tensor
    yc_on_grid: torch.Tensor

    xc_off_grid: torch.Tensor
    yc_off_grid: torch.Tensor


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
        ontg_x = make_grid(
            xmin = self.grid_range[:, 0].repeat(*batch_shape, 1), 
            xmax = self.grid_range[:, 1].repeat(*batch_shape, 1), 
            points_per_unit = self.points_per_unit, 
            margin = 0)
        ontg_x = flatten_grid(ontg_x) # shape (batch, num_ontg, xdim)

        # (batch_shape, num_ctx + num_trg + num_ontg, xdim).
        x = torch.cat((offtg_x, ontg_x), dim=-2)

        # (batch_shape, num_ctx + num_trg + num_ontg, GP_out_dim). GP_out_dim should be 2.
        y, gt_pred = self.otg_generator.sample_outputs(x=x)
        offtg_y = y[:, :offtg_x.shape[-2], :1]
        # Use the other dimension if present, otherwise use same dimension
        ontg_y = y[:, offtg_x.shape[-2]:, 1 if y.shape[-1] > 1 else 0].unsqueeze(-1)

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
            xc_on_grid=ontg_x,
            yc_on_grid=ontg_y,
            xt=xt,
            yt=yt,
            gt_pred=None #TODO re-enable this once it works.
        )