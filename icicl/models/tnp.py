import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TransformerEncoder
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
        yt = torch.zeros(xt.shape[:-1] + yc.shape[-1:])

        x = torch.cat((xc, xt), dim=-2)
        y = torch.cat((yc, yt), dim=-2)
        z = torch.cat((x, y), dim=-1)
        z = self.xy_encoder(z)

        # Construct mask.
        mask = torch.ones(x.shape[:-1] + x.shape[-2:-1]) > 0.5
        mask[:, : xc.shape[-2], : xc.shape[-2]] = False
        for i in range(xt.shape[-2]):
            mask[:, xc.shape[-2] + i, : xc.shape[-2] + i] = False

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
