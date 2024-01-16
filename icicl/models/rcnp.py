from typing import Callable

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.mlp import MLP
from ..utils.group_actions import translation
from .base import NeuralProcess
from .cnp import CNPDecoder


class RCNPEncoder(nn.Module):
    def __init__(
        self,
        relational_encoder: MLP,
        group_action: Callable = translation,
        diagonal_encoding: bool = False,
        agg: str = "sum",
    ):
        super().__init__()

        self.relational_encoder = relational_encoder
        self.group_action = group_action
        self.diagonal_encoding = diagonal_encoding

        if agg == "sum":
            self.agg = (
                lambda x: torch.sum(x, dim=-2)
                if self.diagonal_encoder
                else lambda x: torch.sum(x, dim=(-2, -3))
            )
        elif agg == "mean":
            self.agg = (
                lambda x: torch.mean(x, dim=-2)
                if self.diagonal_encoder
                else lambda x: torch.mean(x, dim=(-2, -3))
            )
        else:
            raise ValueError("agg must be one of 'sum', 'mean'.")

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        # Compute relational matrices.

        # (batch_size, nc, nc, dx).
        dcc = self.group_action(xc, xc)

        # (batch_size, nt, nc, dx).
        dtc = self.group_action(xt, xc)

        if self.diagonal_encoding:
            # Get diagonal elements of dcc.
            # (batch_size, nc, dx).
            dc = torch.diagonal(dcc, dim1=1, dim2=2)
            dc = einops.rearrange(dc, "m d n -> m n d")

            # Construct inputs to relational encoder.
            dc = einops.repeat(dc, "m nc d -> m nt nc d", nt=xt.shape[1])
            yc = einops.repeat(yc, "m nc d -> m nt nc d", nt=xt.shape[1])
            ztc = torch.cat((dtc, dc, yc), dim=-1)

            # (batch_size, nt, nc, dz).
            ztc = self.relational_encoder(ztc)

            # Sum over context points and return.
            return self.agg(ztc)

        # This seems quite insane.
        yc = einops.repeat(yc, "m nc1 d -> m nc1 nc2 d", nc2=xc.shape[1])
        rcc = torch.cat((dcc, yc, yc.transpose(1, 2)), dim=-1)

        # This bit in particular.
        rcc = einops.repeat(rcc, "m nc1 nc2 d -> m nt nc1 nc2 d", nt=xt.shape[1])
        dtc = einops.repeat(dtc, "m nt nc1 d -> m nt nc1 nc2 d", nc2=xc.shape[1])
        ztc = torch.cat((dtc, rcc), dim=-1)

        # (batch_size, nt, nc, nc, dz).
        ztc = self.relational_encoder(ztc)

        # Sum over context points and return.
        return self.agg(ztc)


class RCNP(NeuralProcess):
    def __init__(
        self,
        encoder: RCNPEncoder,
        decoder: CNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
