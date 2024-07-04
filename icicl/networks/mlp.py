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

        if layers is None:
            # Check that one of the two specifications is given.
            assert (
                num_layers is not None and width is not None
            ), "Must specify either `layers` or `num_layers` and `width`."
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
    

class MLPWithEmbedding(MLP):
    def __init__(
        self,
        embedding: nn.Module,
        ignore_dims: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding = embedding
        self.ignore_dims = ignore_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_dims is not None:
            active_dims = [
                dim for dim in range(x.shape[-1]) if dim not in self.ignore_dims
            ]
            x_ignore = x[..., self.ignore_dims]
            x_active = x[..., active_dims]
        else:
            x_ignore = None
            x_active = x

        out = self.embedding(x_active)
        out = super().forward(out)

        if x_ignore is not None:
            out = torch.cat((x_ignore, out), dim=-1)

        return out
