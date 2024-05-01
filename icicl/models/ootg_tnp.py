from typing import Union
import torch
from torch import nn

from .base import OOTGConditionalNeuralProcess
from .tnp import TNPDEncoder, TNPDDecoder, gen_tnpd_mask
from ..utils.conv import compute_eq_weights
from ..utils.helpers import preprocess_observations


class OOTGSetConvEncoder(TNPDEncoder):

    def __init__(
        self,
        dim: int,
        init_lengthscale: float,
        train_lengthscale: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)  
        init_lengthscale = torch.as_tensor(dim * [init_lengthscale], dtype=dtype)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=train_lengthscale,
        )

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )

    def forward(
        self,
        xc_off_grid: torch.Tensor,
        yc_off_grid: torch.Tensor,
        xc_on_grid: torch.Tensor,
        yc_on_grid: torch.Tensor,
        xt: torch.Tensor
    ) -> torch.Tensor:
        weights = compute_eq_weights(xc_on_grid, xc_off_grid, lengthscales=self.lengthscale)
        yc_off_grid_gridded = weights @ yc_off_grid
        
        # shape (batch_size, num_ontg, xdim)
        xc = xc_on_grid
        # shape (batch_size, num_ontg, 2)
        yc = torch.cat((yc_on_grid, yc_off_grid_gridded), dim=-1)

        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=-2)
        y = torch.cat((yc, yt), dim=-2)
        z = torch.cat((x, y), dim=-1)
        z = self.xy_encoder(z)

        # Construct mask.
        mask = gen_tnpd_mask(
            xc,
            xt,
            contexts_self_attend=self.contexts_self_attend,
            contexts_cross_attend=self.contexts_cross_attend,
            targets_self_attend=self.targets_self_attend,
            ar_mode=self.ar_mode,
        )

        z = self.transformer_encoder(z, mask)
        return z


class OOTG_TNPD(OOTGConditionalNeuralProcess):
    def __init__(
        self,
        encoder: OOTGSetConvEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
