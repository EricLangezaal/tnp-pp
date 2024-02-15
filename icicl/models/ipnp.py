from typing import Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import SPINDecoder, SPINEncoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess


class IPNPEncoder(nn.Module):
    def __init__(
        self,
        spin_encoder: SPINEncoder,
        attribute_encoder: nn.Module,
    ):
        super().__init__()

        self.spin_encoder = spin_encoder
        self.attribute_encoder = attribute_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return[0]: [m, nq, dz]",
        "return[1]: [m, nt, dy_]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        _ = xt

        yc, yt = preprocess_observations(xt, yc)

        zc = torch.cat((xc, yc), dim=-1)
        zc = einops.rearrange(zc, "m n d -> (m n d) 1")
        zc = self.attribute_encoder(zc)
        zc = einops.rearrange(
            zc,
            "(m n d) e -> m n d e",
            m=xc.shape[0],
            n=xc.shape[1],
            d=xc.shape[-1] + yc.shape[-1],
        )

        zq = self.spin_encoder(zc)
        return zq, yt


class IPNPDecoder(nn.Module):
    def __init__(
        self,
        spin_decoder: SPINDecoder,
        attribute_encoder: nn.Module,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.spin_decoder = spin_decoder
        self.attribute_encoder = attribute_encoder
        self.z_decoder = z_decoder

    @check_shapes(
        "out[0]: [m, nq, dz]",
        "out[1]: [m, nt, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dy]",
    )
    def forward(
        self, out: Tuple[torch.Tensor, torch.Tensor], xt: torch.Tensor
    ) -> torch.Tensor:
        zq, yt_ = out

        zt = torch.cat((xt, yt_), dim=-1)
        zt = einops.rearrange(zt, "m n d -> (m n d) 1")
        zt = self.attribute_encoder(zt)
        zt = einops.rearrange(
            zt,
            "(m n d) e -> m n d e",
            m=xt.shape[0],
            n=xt.shape[1],
            d=xt.shape[-1] + yt_.shape[-1],
        )

        zt = self.spin_decoder(zt, zq)
        return self.z_decoder(zt)


class IPNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: IPNPEncoder,
        decoder: IPNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
