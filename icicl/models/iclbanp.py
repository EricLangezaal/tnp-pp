import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import ICNestedPerceiverEncoder
from ..utils.helpers import preprocess_observations
from .base import ICNeuralProcess
from .lbanp import NestedLBANPDecoder


class NestedICLBANPEncoder(nn.Module):
    def __init__(
        self,
        perceiver_encoder: ICNestedPerceiverEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.perceiver_encoder = perceiver_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xic: [m, nic, ncic, dx]",
        "yic: [m, nic, ncic, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nq, dz]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xic: torch.Tensor,
        yic: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)
        yic = torch.cat((yic, torch.zeros(yic.shape[:-1] + (1,)).to(yic)), dim=-1)

        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zic = torch.cat((xic, yic), dim=-1)
        zic = self.xy_encoder(zic)

        zt = torch.cat((xt, yt), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.perceiver_encoder(zc, zic, zt)
        return zt


class ICLBANP(ICNeuralProcess):
    def __init__(
        self,
        encoder: NestedICLBANPEncoder,
        decoder: NestedLBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
