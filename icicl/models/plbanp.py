from torch import nn

from ..networks.transformer import ParallelNestedPerceiverEncoder
from .base import NeuralProcess
from .lbanp import NestedLBANPDecoder, NestedLBANPEncoder


class NestedPLBANPEncoder(NestedLBANPEncoder):
    def __init__(
        self,
        perceiver_encoder: ParallelNestedPerceiverEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__(
            perceiver_encoder=perceiver_encoder,
            xy_encoder=xy_encoder,
        )


class PLBANP(NeuralProcess):
    def __init__(
        self,
        encoder: NestedPLBANPEncoder,
        decoder: NestedLBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
