import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TransformerEncoder
from ..utils.helpers import preprocess_observations
from .base import NeuralProcess


class TNPDEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TransformerEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=-2)
        y = torch.cat((yc, yt), dim=-2)
        z = torch.cat((x, y), dim=-1)
        z = self.xy_encoder(z)

        # Construct mask.
        mask = gen_tnpd_mask(xc, xt, targets_self_attend=True)

        z = self.transformer_encoder(z, mask)
        return z


class TNPDDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("z: [m, n, dz]", "xt: [m, nt, dx]", "return: [m, nt, dy]")
    def forward(self, z: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        zt = z[:, -xt.shape[-2] :, ...]
        return self.z_decoder(zt)


class TNPD(NeuralProcess):
    def __init__(
        self,
        encoder: TNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


@check_shapes("xc: [m, nc, dx]", "xt: [m, nt, dx]", "return: [m, n, n]")
def gen_tnpd_mask(
    xc: torch.Tensor, xt: torch.Tensor, targets_self_attend: bool = False
) -> torch.Tensor:
    m = xc.shape[0]
    nc = xc.shape[-2]
    nt = xt.shape[-2]
    n = nc + nt

    mask = torch.ones(m, n, n) > 0.5
    mask[:, :nc, :nc] = False
    for i in range(xt.shape[-2]):
        mask[:, nc + i, :nc] = False

        if targets_self_attend:
            mask[:, nc + i, nc + i] = False

    return mask
