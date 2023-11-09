import torch
from check_shapes import check_shapes


@check_shapes("x1: [m, n1, dx]", "x2: [m, n2, dx]", "return: [m, n1, n2, dx]")
def translation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1[:, :, None, :] - x2[:, None, :, :]


@check_shapes("x1: [m, n1, dx]", "x2: [m, n2, dx]", "return: [m, n1, n2, dout]")
def rotation_1d(
    x1: torch.Tensor, x2: torch.Tensor, xmin: float, xmax: float
) -> torch.Tensor:
    """Applies rotation equivariance to each dimension independently.

    Args:
        x1 (torch.Tensor)
        x2 (torch.Tensor)
        xmin (float): Minimum x value.
        xmax (float): Maximum x value.

    Returns:
        torch.Tensor: Differences in sine / cosine values.
    """
    x1_ = 2 * torch.pi * (x1[..., 0] - xmin) / (xmax - xmin)
    x2_ = 2 * torch.pi * (x2[..., 0] - xmin) / (xmax - xmin)

    x1_ = torch.cat((torch.sin(x1_), torch.cos(x1_)), dim=-1)
    x2_ = torch.cat((torch.sin(x2_), torch.cos(x2_)), dim=-1)

    return translation(x1_, x2_)
