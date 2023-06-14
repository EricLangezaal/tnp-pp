from typing import Tuple

import mlkernels
import numpy as np
import stheno
import torch


def batch(x, other_dims):
    """Get the shape of the batch of a tensor.

    Args:
        x (tensor): Tensor.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tuple[int]: Shape of batch dimensions.
    """
    return x.size()[:-other_dims]


def compress_batch_dimensions(x, other_dims):
    """Compress multiple batch dimensions of a tensor into a single batch dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tensor: `x` with batch dimensions compressed.
        function: Function to undo the compression of the batch dimensions.
    """
    b = batch(x, other_dims)
    if len(b) == 1:
        return x, lambda x: x
    else:

        def uncompress(x_after):
            return torch.reshape(x_after, (*b, *x_after.size()[1:]))

        return (
            torch.reshape(x, (int(np.prod(b)), *x.size()[len(b) :])),
            uncompress,
        )


def gp_dataset_generator(
    x_min: float = -3.0,
    x_max: float = 3.0,
    min_n: int = 60,
    max_n: int = 120,
    noise: float = 0.05,
    kernel: mlkernels.Kernel = stheno.EQ(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_min < x_max
    assert min_n < max_n

    gp = stheno.GP(kernel)

    # Randomly sample input points from range.
    n = torch.randint(low=min_n, high=max_n, size=(1,))
    x = torch.rand((n, 1)) * (x_min - x_max) + x_max
    y = gp(x, noise).sample()

    return x, y
