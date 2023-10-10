import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.tetransformer import TETransformerEncoder
from .base import NeuralProcess
from .tnp import TNPDDecoder, gen_tnpd_mask


class TETNPDEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: TETransformerEncoder,
        y_encoder: nn.Module,
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        yt = torch.zeros(xt.shape[:-1] + yc.shape[-1:])
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,))), dim=-1)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,))), dim=-1)

        x = torch.cat((xc, xt), dim=-2)
        y = torch.cat((yc, yt), dim=-2)
        z = self.y_encoder(y)

        # Construct mask.
        mask = gen_tnpd_mask(xc, xt, targets_self_attend=True)

        z = self.transformer_encoder(x, z, mask)
        return z


class TETNPD(NeuralProcess):
    def __init__(
        self,
        encoder: TETNPDEncoder,
        decoder: TNPDDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
