import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import NestedPerceiverEncoder
from .base import NeuralProcess


class LBANPEncoder(nn.Module):
    def __init__(
        self,
        nested_perceiver_encoder: NestedPerceiverEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__()

        self.nested_perceiver_encoder = nested_perceiver_encoder
        self.xy_encoder = xy_encoder

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, nq, dz]"
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        zc = torch.cat((xc, yc), dim=-1)
        zc = self.xy_encoder(zc)

        zt = torch.cat((xt, torch.zeros(*xt.shape[:-1], yc.shape[-1])), dim=-1)
        zt = self.xy_encoder(zt)

        zt = self.nested_perceiver_encoder(zc, zt)
        return zt


class LBANPDecoder(nn.Module):
    def __init__(
        self,
        z_decoder: nn.Module,
    ):
        super().__init__()

        self.z_decoder = z_decoder

    @check_shapes("zt: [m, nt, dz]", "xt: [m, nt, dx]", "return: [m, nt, dy]")
    def forward(
        self,
        zt: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        _ = xt

        return self.z_decoder(zt)


class LBANP(NeuralProcess):
    def __init__(
        self,
        encoder: LBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)


class LBANPlusEncoder(LBANPEncoder):
    def __init__(
        self,
        nested_perceiver_encoder: NestedPerceiverEncoder,
        xy_encoder: nn.Module,
        embedding_token_dim: int,
        dy: int = 1,
    ):
        super().__init__(nested_perceiver_encoder, xy_encoder)

        self.context_embedding_token = nn.Parameter(torch.randn(embedding_token_dim))
        self.target_embedding_token = nn.Parameter(
            torch.randn(embedding_token_dim + dy)
        )

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        ec = einops.repeat(
            self.context_embedding_token, "e -> m n e", m=xc.shape[0], n=xc.shape[1]
        )
        zc = torch.cat((xc, yc, ec), dim=-1)
        zc = self.xy_encoder(zc)

        et = einops.repeat(
            self.target_embedding_token, "e -> m n e", m=xt.shape[0], n=xt.shape[1]
        )
        zt = torch.cat((xc, et), dim=-1)
        zt = self.xy_encoder(zt)

        zq = self.nested_perceiver_encoder(zc, zt)
        return zq


class LBANPlus(NeuralProcess):
    def __init__(
        self,
        encoder: LBANPlusEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
