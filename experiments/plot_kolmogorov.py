import os
from typing import Callable, List, Tuple, Union

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import wandb
from icicl.data.kolmogorov import KolmogorovBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

plot_dims = [0, 1]
time_dim = [2]


def compute_vorticity(
    v: torch.Tensor, batch_grid_size: Tuple[int, int, int], time_slice: int
):
    v = einops.rearrange(
        v,
        "1 (h w t) d -> d h w t",
        h=batch_grid_size[0],
        w=batch_grid_size[1],
        t=batch_grid_size[2],
    )

    # Select point in time we want to plot.
    v = v[..., time_slice]

    # Compute derivatives.
    (du,) = torch.gradient(v[0], dim=-1)
    (dv,) = torch.gradient(v[1], dim=-2)

    w = du - dv
    w = w[1:-1, 1:-1]

    return w


def plot_kolmogorov(
    model: Union[
        nn.Module,
        Callable[..., torch.distributions.Distribution],
    ],
    batches: List[KolmogorovBatch],
    batch_grid_size: Tuple[int, int, int] = (16, 16, 16),
    time_slice: int = 0,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (18.0, 5.0),
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    subplots: bool = True,
):
    for i in range(num_fig):
        batch = batches[i]
        x = batch.x[:1]
        y = batch.y[:1]
        xc = batch.xc[:1]
        yc = batch.yc[:1]
        xt = batch.xt[:1]
        yt = batch.yt[:1]
        re = batch.re[0]

        if not isinstance(model, nn.Module):
            # model is callable function that trains model then returns pred.
            y_pred_dist = model(xc=xc, yc=yc, xt=x)
            yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

            # Detach gradients.
            y_pred_dist.loc = y_pred_dist.loc.detach().unsqueeze(0)
            y_pred_dist.scale = y_pred_dist.scale.detach().unsqueeze(0)
            yt_pred_dist.loc = yt_pred_dist.loc.detach().unsqueeze(0)
            yt_pred_dist.scale = yt_pred_dist.scale.detach().unsqueeze(0)
        else:
            with torch.no_grad():
                y_pred_dist = model(xc=xc, yc=yc, xt=x)
                yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

        # model_nll = -yt_pred_dist.log_prob(yt).mean().cpu()
        mean = y_pred_dist.loc.cpu()
        # std = y_pred_dist.scale.cpu()

        # Compute vorcitiy.
        gt_v = compute_vorticity(y.cpu(), batch_grid_size, time_slice)
        pred_v = compute_vorticity(mean.cpu(), batch_grid_size, time_slice)

        vmin = gt_v.min()
        vmax = gt_v.max()

        pc = xc.numel() / x.numel()

        # Create random mask to fake context.
        mc = np.random.rand(*gt_v.shape)
        ctx_v = np.ma.masked_where(mc > pc, gt_v)

        if subplots:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

            for v, ax, fig_name in zip(
                (ctx_v, gt_v, pred_v),
                axes,
                ("Fake context", "Grounth truth", "Pred mean"),
            ):
                ax.imshow(v, vmin=vmin, vmax=vmax, cmap="inferno")
                ax.set_title(fig_name)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])

            plt.suptitle(f"Reynolds: {re}")

            fname = f"fig/{name}/{i:03d}-pc-{pc:2f}"
            if wandb.run is not None and logging:
                wandb.log({fname: wandb.Image(fig)})
            elif savefig:
                if not os.path.isdir(f"fig/{name}"):
                    os.makedirs(f"fig/{name}")
                plt.savefig(fname, bbox_inches="tight")
            else:
                plt.show()

            plt.close()

        else:
            for v, fig_name in zip(
                (ctx_v, gt_v, pred_v), ("Fake context", "Grounth truth", "Pred mean")
            ):
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                ax.imshow(v, vmin=vmin, vmax=vmax, cmap="inferno")
                # ax.set_title(name)
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])

                fname = f"fig/{name}/{i:03d}-pc-{pc:2f}/{fig_name}"
                if wandb.run is not None and logging:
                    wandb.log({fname: wandb.Image(fig)})
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}-pc-{pc:2f}"):
                        os.makedirs(f"fig/{name}/{i:03d}-pc-{pc:2f}")
                    plt.savefig(fname, bbox_inches="tight")
                else:
                    plt.show()

                plt.close()

    # # Select data to plot.
    # time_vals = x[0, :, time_dim].unique()
    # time_val = time_vals[time_slice]

    # data_idx = torch.where(x[0, ..., time_dim] == time_val)[0].cpu()
    # data_idx_c = torch.where(xc[0, ..., time_dim] == time_val)[0].cpu()

    # x_ = x[0, data_idx].cpu()
    # y_ = y[0, data_idx].cpu()
    # xc_ = xc[0, data_idx_c].cpu()
    # yc_ = yc[0, data_idx_c].cpu()
    # mean_ = mean[0, data_idx].cpu()
    # std_ = std[0, data_idx].cpu()

    # # vmin = min(y_.min(), mean_.min())
    # # vmax = max(y_.max(), mean_.max())
    # vmin = y_.min()
    # vmax = y_.max()

    # scatter_kwargs = {
    #     "s": 45,
    #     "marker": "s",
    #     "alpha": 1.0,
    #     "vmin": vmin,
    #     "vmax": vmax,
    #     "lw": 0,
    # }

    # xmin = x_[:, plot_dims[0]].min()
    # xmax = x_[:, plot_dims[0]].max()
    # ymin = x_[:, plot_dims[1]].min()
    # ymax = x_[:, plot_dims[1]].max()
    # xlim = (xmin, xmax)
    # ylim = (ymin, ymax)

    # if subplots:
    #     fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=1)

    #     for i, (x_plot, y_plot) in enumerate(zip((xc_, x_, x_), (yc_, y_, mean_))):
    #         v = (y_plot - scatter_kwargs["vmin"]) / (
    #             scatter_kwargs["vmax"] - scatter_kwargs["vmin"]
    #         )
    #         v = 2 * v + 1
    #         v = torch.sign(v) * torch.abs(v) ** 0.8
    #         v = (v + 1) / 2

    #         ax = axes[i]
    #         sc = ax.scatter(
    #             x_plot[:, plot_dims[0]],
    #             x_plot[:, plot_dims[1]],
    #             c=v,
    #             **scatter_kwargs,
    #         )
    #         ax.set_aspect("equal")
    #         ax.set_xlim(xlim)
    #         ax.set_ylim(ylim)

    #     axes[0].set_title("Context set", fontsize=18)
    #     axes[1].set_title("Ground truth", fontsize=18)
    #     axes[2].set_title("Predictive mean", fontsize=18)

    #     # Add colourbar.
    #     cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
    #     cbar.solids.set(alpha=1)

    #     plt.suptitle(
    #         f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}    "
    #         #
    #         f"NLL = {model_nll:.3f}",
    #         fontsize=18,
    #     )

    #     fname = f"fig/{name}/{i:03d}"
    #     if wandb.run is not None and logging:
    #         wandb.log({fname: wandb.Image(fig)})
    #     elif savefig:
    #         if not os.path.isdir(f"fig/{name}"):
    #             os.makedirs(f"fig/{name}")
    #         plt.savefig(fname)
    #     else:
    #         plt.show()

    # else:
    #     for y_dim in range(y.shape[-1]):
    #         for fig_name, x_plot, y_plot in zip(
    #             ("context", "ground_truth", "pred_mean", "pred_std"),
    #             (xc_, x_, x_, x_),
    #             (yc_, y_, mean_, std_),
    #         ):
    #             fig = plt.figure(figsize=figsize)

    #             ax = plt.gca()
    #             if fig_name == "pred_std":
    #                 std_scatter_kwargs = scatter_kwargs
    #                 std_scatter_kwargs["vmin"] = y_plot.min()
    #                 std_scatter_kwargs["vmax"] = y_plot.max()
    #                 sc = ax.scatter(
    #                     x_plot[:, plot_dims[0]],
    #                     x_plot[:, plot_dims[1]],
    #                     c=y_plot[:, y_dim],
    #                     **std_scatter_kwargs,
    #                 )
    #             else:
    #                 sc = ax.scatter(
    #                     x_plot[:, plot_dims[0]],
    #                     x_plot[:, plot_dims[1]],
    #                     c=y_plot[:, y_dim],
    #                     **scatter_kwargs,
    #                 )

    #             ax.set_aspect("equal")
    #             ax.set_xlim(xlim)
    #             ax.set_ylim(ylim)

    #             fname = f"fig/{name}/{i:03d}/{fig_name}/ydim-{y_dim}"
    #             if wandb.run is not None and logging:
    #                 wandb.log({fname: wandb.Image(fig)})
    #             elif savefig:
    #                 if not os.path.isdir(f"fig/{name}/{i:03d}/{fig_name}"):
    #                     os.makedirs(f"fig/{name}/{i:03d}/{fig_name}")
    #                 plt.savefig(fname)
    #             else:
    #                 plt.show()

    #             plt.close()
