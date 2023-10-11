from torch import nn

from ..networks.transformer import ParallelNestedPerceiverEncoder
from .base import NeuralProcess
from .lbanp import LBANPDecoder, LBANPEncoder


class PLBANPEncoder(LBANPEncoder):
    def __init__(
        self,
        parallel_nested_perceiver_encoder: ParallelNestedPerceiverEncoder,
        xy_encoder: nn.Module,
    ):
        super().__init__(
            nested_perceiver_encoder=parallel_nested_perceiver_encoder,
            xy_encoder=xy_encoder,
        )


class PLBANP(NeuralProcess):
    def __init__(
        self,
        encoder: PLBANPEncoder,
        decoder: LBANPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
