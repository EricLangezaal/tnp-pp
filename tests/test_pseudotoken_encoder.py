import torch
import einops
from torch.nn.attention import SDPBackend, sdpa_kernel

from icicl.data.on_off_grid import DataModality
from icicl.utils.grids import unflatten_grid, flatten_grid
from icicl.networks.grid_encoders import PseudoTokenGridEncoder
from icicl.networks.attention_layers import MultiHeadCrossAttentionLayer


def test_pt_grid_encoder(dim:int = 1, emb_dim=128):
    x_off_grid = torch.linspace(-1, 1, 10)[None, :, None].repeat(1,1,dim)
    z_off_grid = torch.randn(x_off_grid.shape[:-1] + (emb_dim,))
    x_on_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.linspace(-1, 1, steps=8, dtype=torch.float) for _ in range(dim)
                ],
                indexing="ij"
            ),
            dim=-1,
        )[None, ...]
    z_on_grid = torch.randn(x_on_grid.shape[:-1] + (emb_dim,))
    assert x_on_grid.dim() == dim + 2

    with torch.no_grad():
        layer = MultiHeadCrossAttentionLayer(embed_dim=emb_dim, num_heads=8, head_dim=16, feedforward_dim=emb_dim)

        encoder = PseudoTokenGridEncoder(embed_dim=emb_dim, mhca_layer=layer,grid_range=[[-1,1]]* dim, points_per_unit= 8 // 2)

        _, out_current = encoder(x_off_grid, x_on_grid, z_off_grid, z_on_grid, used_modality=DataModality.BOTH)
        out_old = old_pt_grid_encoder(layer, encoder.latents.view(-1, emb_dim), x_off_grid, x_on_grid, z_off_grid, z_on_grid)

    assert torch.equal(out_current, out_old)


def old_pt_grid_encoder(
        layer: MultiHeadCrossAttentionLayer,
        latents: torch.Tensor,
        x_off_grid: torch.Tensor, 
        x_on_grid: torch.Tensor,  
        z_off_grid: torch.Tensor, 
        z_on_grid: torch.Tensor
) -> torch.Tensor:
    # ---------------------- calculate output manually using old implementation -----------------------------------

    grid_shape = x_on_grid.shape[1:-1]
    x_on_grid = flatten_grid(x_on_grid)
    z_on_grid = flatten_grid(z_on_grid)

    B, U, E = z_off_grid.shape # 'U'nstructured
    S = z_on_grid.shape[-2] # 'S'tructured
    nearest_idx = (
        (x_off_grid[..., None, :] - x_on_grid[:, None, ...]).abs().sum(dim=-1).argmin(dim=2)
    )
    s_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, S)
    u_batch_idx = torch.arange(B).unsqueeze(-1).repeat(1, U)
    s_range_idx = torch.arange(S).repeat(B, 1)
    u_range_idx = torch.arange(U).repeat(B, 1)
    nearest_mask = torch.zeros(B, U, S, dtype=torch.int)
    nearest_mask[u_batch_idx, u_range_idx, nearest_idx] = 1
    
    max_patch = nearest_mask.sum(dim=1).amax() + 1
    cumcount_idx = (nearest_mask.cumsum(dim=1) - 1)[u_batch_idx, u_range_idx, nearest_idx]

    joint_grid = torch.full((B, S, max_patch, E), -torch.inf)
    joint_grid[u_batch_idx, nearest_idx, cumcount_idx] = z_off_grid
    joint_grid[s_batch_idx, s_range_idx, -1] = z_on_grid
    grid_stacked = einops.rearrange(joint_grid, "b s m e -> (b s) m e")

    att_mask = torch.ones(B * S, 1, max_patch, dtype=torch.bool)
    att_mask[(grid_stacked.sum(-1) == -torch.inf).unsqueeze(1)] = False
    grid_stacked[grid_stacked == -torch.inf] = 0

    latents = einops.repeat(latents, "s e -> (b s) 1 e", b=B)
    with sdpa_kernel(SDPBackend.MATH):
        z = layer(latents, grid_stacked, mask=att_mask)
    z = einops.rearrange(z, "(b s) 1 e -> b s e", b=B)
    return unflatten_grid(z, grid_shape)

if __name__ == "__main__":
    test_pt_grid_encoder(dim=1)
    test_pt_grid_encoder(dim=2)