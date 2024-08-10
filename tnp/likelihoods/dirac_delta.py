import torch
from torch import nn
import torch.distributions as td

from .base import Likelihood

class CustomDelta(td.Distribution):
    def __init__(self, output, **kwargs):
        super().__init__(validate_args=False, **kwargs)
        self.output = output

    @property
    def mean(self):
        return self.output
    
    @property
    def stddev(self):
        return torch.zeros_like(self.output)
    
    def log_prob(self, value):
       return -nn.functional.mse_loss(self.output, value, reduction='none')


class DeltaLikelihood(Likelihood):

    def forward(self, x: torch.Tensor) -> CustomDelta:
        return CustomDelta(x)
