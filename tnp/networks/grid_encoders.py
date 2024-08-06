from typing import Tuple, Union, Optional, Callable

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import einops
from check_shapes import check_shapes

from ..data.on_off_grid import DataModality
from ..networks.attention_layers import MultiHeadCrossAttentionLayer
from ..networks.attention import MultiHeadCrossAttention
from ..utils.conv import compute_eq_weights
from ..utils.grids import unflatten_grid, flatten_grid, make_grid_from_range, nearest_gridded_neighbours, coarsen_grid

class IdentityGridEncoder(nn.Module):

    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]",
        "return[0]: [b, t, dx]", "return[1]: [b, t, dz]"
    )
    def forward(
        self, 
        xc_off_grid: torch.Tensor, 
        xc_on_grid: torch.Tensor, 
        zc_off_grid: torch.Tensor, 
        zc_on_grid: torch.Tensor, 
        used_modality: DataModality,
    ) -> torch.Tensor:
        if used_modality == DataModality.OFF_GRID:
            return xc_off_grid, zc_off_grid
        elif used_modality == DataModality.ON_GRID:
            return flatten_grid(xc_on_grid), flatten_grid(zc_on_grid)
        
        xc = torch.cat((xc_off_grid, flatten_grid(xc_on_grid)), dim=-2)
        zc = torch.cat((zc_off_grid, flatten_grid(zc_on_grid)), dim=-2)
        return xc, zc
    

class SetConvGridEncoder(IdentityGridEncoder):

    def __init__(
        self,
        *,
        ard_num_dims: Optional[int] = None,
        init_lengthscale: float,
        grid_shape: Optional[Tuple[int, ...]] = None,
        coarsen_fn: Callable = coarsen_grid,
        dist_fn: Optional[Callable] = None,
    ):
        super().__init__() 
        self.ard_num_dims = ard_num_dims
        num_lengthscale_dims = 1 if ard_num_dims is None else ard_num_dims
        
        init_lengthscale = torch.as_tensor(num_lengthscale_dims * [init_lengthscale], dtype=torch.float32)
        self.lengthscale_param = nn.Parameter(
            (init_lengthscale.clone().detach().exp() - 1).log(),
            requires_grad=True,
        )
        self.grid_shape = None if grid_shape is None else torch.as_tensor(grid_shape)
        self.coarsen_fn = coarsen_fn
        self.dist_fn = dist_fn

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )
    
    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]",
        "return[0]: [b, ..., dx]", "return[1]: [b, ..., dz]"
    )
    def forward(
        self, 
        xc_off_grid: torch.Tensor, 
        xc_on_grid: torch.Tensor, 
        zc_off_grid: torch.Tensor, 
        zc_on_grid: torch.Tensor, 
        used_modality: DataModality,
    ) -> torch.Tensor:
        
        if self.grid_shape is None:
            assert used_modality != DataModality.ON_GRID, (
                "Without grid coarsening, an IdentityGridEncoder has to be used to ignore off-grid data."
            )
            x_grid = xc_on_grid
            z_grid = setconv_to_grid(
                x=xc_off_grid, 
                z=zc_off_grid, 
                x_grid=x_grid, 
                lengthscale=self.lengthscale,
                z_grid=None if used_modality == DataModality.OFF_GRID else zc_on_grid, 
                dist_func=self.dist_fn)
        else:
            xc, zc = super().forward(xc_off_grid, xc_on_grid, zc_off_grid, zc_on_grid, used_modality)

            grid_shape = torch.as_tensor(xc_on_grid.shape[1:-1])
            assert torch.all(grid_shape % self.grid_shape == 0), (
                "cannot properly coarsen incoming grid to match pseudo-grid."
                )
            x_grid = self.coarsen_fn(xc_on_grid, (grid_shape // self.grid_shape).to(int).tolist())

            z_grid = setconv_to_grid(x=xc, z=zc, x_grid=x_grid, lengthscale=self.lengthscale, dist_func=self.dist_fn)
        return x_grid, z_grid
    

@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def setconv_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    lengthscale: torch.Tensor,
    z_grid: Optional[torch.Tensor] = None,
    dist_func: Optional[Callable] = None,
):
    grid_shape = x_grid.shape[1:-1]
    x_grid_flat = flatten_grid(x_grid)
    weights = compute_eq_weights(x_grid_flat, x, lengthscale, dist_func)

    # Multiply context outputs by weights.
    # (batch_size, num_grid_points, embed_dim).
    z_grid_flat = weights @ z

    # Reshape grid.
    if z_grid is None:
        z_grid = 0

    z_grid = z_grid + unflatten_grid(z_grid_flat, grid_shape)

    return z_grid
    

class PseudoTokenGridEncoder(IdentityGridEncoder):
    FAKE_TOKEN = -torch.inf

    def __init__(
            self,
            *,
            embed_dim: int,
            mhca_layer: Union[MultiHeadCrossAttentionLayer, MultiHeadCrossAttention],
            grid_shape: Optional[Tuple[int, ...]] = None, 
            grid_range: Optional[Tuple[Tuple[float, float], ...]] = None,
            points_per_unit: Optional[int] = None,
            coarsen_fn: Callable = coarsen_grid,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        if self.grid_shape is None:
            assert grid_range is not None and points_per_unit is not None, (
                "either specify a grid_shape or a grid_range and points_per_unit."
            )
            self.grid_shape = make_grid_from_range(grid_range, points_per_unit).shape[:-1]

        self.latents = nn.Parameter(torch.randn(*self.grid_shape, embed_dim))
        self.grid_shape = torch.as_tensor(self.grid_shape)
        self.coarsen_fn = coarsen_fn

        self.fake_embedding = nn.Parameter(torch.randn(embed_dim))
        self.mhca_layer = mhca_layer

    @check_shapes(
        "xc_off_grid: [b, n, dx]", "xc_on_grid: [b, ..., dx]", "zc_off_grid: [b, n, dz]", "zc_on_grid: [b, ..., dz]",
         "return[0]: [b, ..., dx]", "return[1]: [b, ..., dz]"
    )
    def forward(
            self, 
            xc_off_grid: torch.Tensor, 
            xc_on_grid: torch.Tensor, 
            zc_off_grid: torch.Tensor, 
            zc_on_grid: torch.Tensor, 
            used_modality: DataModality,
    ) -> torch.Tensor:
        grid_shape = torch.as_tensor(xc_on_grid.shape[1:-1])
        latents = self.latents.expand(xc_on_grid.shape[0], *self.latents.shape)

        if torch.equal(grid_shape, self.grid_shape):
            assert used_modality != DataModality.ON_GRID, (
                "Without grid coarsening, an IdentityGridEncoder has to be used to ignore off-grid data."
            )
            # do not coarsen grid.
            z_grid = mhca_to_grid(
                x=xc_off_grid, 
                z=zc_off_grid, 
                x_grid=xc_on_grid, 
                z_grid=zc_on_grid, 
                latent_grid=latents, 
                mhca=self.mhca_layer,
                fake_embedding=self.fake_embedding if used_modality == DataModality.OFF_GRID else None,
            )
            return xc_on_grid, z_grid
        
        else:
            # coarsen grid by having smaller pseudogrid for neighbour mhca
            # concat both modalities or just return either separately depending on ignoring.
            xc, zc = super().forward(xc_off_grid, xc_on_grid, zc_off_grid, zc_on_grid, used_modality)
            
            assert torch.all(grid_shape % self.grid_shape == 0), (
                f"cannot properly coarsen incoming grid of {grid_shape} to match pseudo-grid {self.grid_shape}."
                )
            x_grid = self.coarsen_fn(xc_on_grid, (grid_shape // self.grid_shape).to(int).tolist())
            z_grid = mhca_to_grid(
                x=xc, 
                z=zc, 
                x_grid=x_grid, 
                z_grid=latents, 
                latent_grid=latents, 
                mhca=self.mhca_layer,
                fake_embedding=None,
            )
            return x_grid, z_grid


def mhca_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    z_grid: torch.Tensor,
    latent_grid: torch.Tensor,
    mhca: Union[MultiHeadCrossAttention, MultiHeadCrossAttentionLayer],
    fake_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, U, E = z.shape # 'B'atch, 'U'nstructured, 'E'mbedding

    grid_shape = x_grid.shape[1:-1]
    x_grid_flat = flatten_grid(x_grid)
    z_grid_flat = flatten_grid(z_grid)
    S = x_grid_flat.shape[-2] # 'S'tructured

    # Quick calculation of nearest grid neighbour.
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    # shape (B, U)
    # _batch_ first repeats batch number then increments, range first increments then repeats
    u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
    u_range_idx = torch.arange(U).repeat(B, 1)

    _, inverse_indices = torch.unique(nearest_idx, return_inverse=True)

    sorted_indices = nearest_idx.argsort(dim=1, stable=True)
    inverse_indices_sorted = inverse_indices.gather(1, sorted_indices).type(torch.long)
    unsorted_indices = sorted_indices.argsort(dim=1, stable=True)

    # Store changes in value.
    inverse_indices_diff = inverse_indices_sorted - inverse_indices_sorted.roll(
        1, dims=1
    )
    inverse_indices_diff = torch.where(
        inverse_indices_diff == 0,
        torch.ones_like(inverse_indices_diff),
        torch.zeros_like(inverse_indices_diff),
    )
    inverse_indices_diff[:, 0] = torch.zeros_like(inverse_indices_diff[:, 0])

    adjusted_cumsum = associative_scan(
        inverse_indices_diff, inverse_indices_diff, dim=1
    )
    adjusted_cumsum = adjusted_cumsum.round().int()
    cumcount_idx = adjusted_cumsum.gather(1, unsorted_indices)

    max_patch = cumcount_idx.amax() + 2

    # create tensor with for each grid-token all nearest off-grid + itself in third axis
    grid_stacked = torch.full((B * S, max_patch, E), -torch.inf, device=z_grid.device)

    # add nearest off the grid points to each on_the_grid point
    idx_shifter = torch.arange(0, B * S, S, device=z.device).repeat_interleave(U)
    grid_stacked[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = (
            z[u_batch_idx.flatten(), u_range_idx.flatten()]
    )
    
    if fake_embedding is not None:
        # add learned 'mask out' embedding at the end. Done instead of masking out in att_mask,
        # since that could potentially make it mask out whole input which crashes attention
        grid_stacked[torch.arange(B * S), -1] = fake_embedding
    else:
        # add the on_the_grid points themselves at the end
        s_batch_idx = torch.arange(B).repeat_interleave(S)
        s_range_idx = torch.arange(S).repeat(B)
        grid_stacked[torch.arange(B * S), -1] = z_grid_flat[s_batch_idx, s_range_idx]

    att_mask = torch.ones(B * S, 1, max_patch, dtype=torch.bool, device=z_grid.device)
    # if fake token anywhere (sum with infinities stays infinity) in embedding, ignore it
    att_mask[(grid_stacked.sum(-1) == -torch.inf).unsqueeze(1)] = False
    # set fake value to something which won't overflow/NaN attention calculation
    grid_stacked[grid_stacked == -torch.inf] = 0

    # make latents of shape (B * S, 1, E)
    latent_grid_flat = flatten_grid(latent_grid)
    latent_grid_flat = einops.rearrange(latent_grid_flat, "b m e -> (b m) 1 e")
    
    with sdpa_kernel(SDPBackend.MATH):  
        out_grid_flat = mhca(latent_grid_flat, grid_stacked, mask=att_mask)
    # Reshape output.
    out_grid_flat = einops.rearrange(out_grid_flat, "(b s) 1 e -> b s e", b=B)
    return unflatten_grid(out_grid_flat, grid_shape)


def complex_log(float_input: torch.Tensor, eps=1e-6) -> torch.ComplexType:
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi

    return torch.complex(real, imag)


def associative_scan(
    values: torch.Tensor, coeffs: torch.Tensor, dim: int
) -> torch.Tensor:
    log_values = complex_log(values.float())
    log_coeffs = complex_log(coeffs.float())
    a_star = torch.cumsum(log_coeffs, dim=dim)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
    log_x = a_star + log_x0_plus_b_star

    return torch.exp(log_x).real
