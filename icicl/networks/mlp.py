from typing import Optional, Tuple

import torch
from torch import nn

from ..utils.batch import compress_batch_dimensions


class MLP(nn.Module):
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Optional[nn.Module] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Check that one of the two specifications is given.
        layers_given = layers is not None
        num_layers_given = num_layers is not None and width is not None
        if not (layers_given or num_layers_given):
            raise ValueError(
                "Must specify either `layers` or `num_layers` and `width`."
            )

        # Make sure that `layers` is a tuple of various widths.
        if not layers_given and num_layers_given:
            layers = (width,) * num_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, 2)
        x = self.net(x)
        x = uncompress(x)
        return x
