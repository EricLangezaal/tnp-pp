import copy
import os
from typing import Callable, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from torch import nn

import wandb
from icicl.data.on_off_grid import OOTGBatch
from icicl.utils.grids import flatten_grid
from icicl.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_era5(
    model: nn.Module,
    batches: List[OOTGBatch],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    grid_idx: int = 1,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    colorbar: bool = True,
    pred_fn: Callable = np_pred_fn,
):

    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        if batch.x is not None and batch.y is not None:
            x_grid, y_grid = batch.x, batch.y
        else:
            x_grid, y_grid = batch.xc_on_grid, batch.yc_on_grid

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = flatten_grid(x_grid)

        with torch.no_grad():
            yt_pred_dist = pred_fn(model, batch)
            pred_mean_t, pred_std_t = yt_pred_dist.mean.cpu(), yt_pred_dist.stddev.cpu()
           
            yplot_pred_dist = pred_fn(model, plot_batch)
            pred_mean_grid, pred_std_grid = (
                yplot_pred_dist.mean.cpu(),
                yplot_pred_dist.stddev.cpu(),
            )

        y_mean_off, y_std_off = y_mean[grid_idx - 1], y_std[grid_idx - 1]
        y_mean_on, y_std_on = y_mean[grid_idx], y_std[grid_idx]
        # Rescale inputs and outputs.
        xc_off_grid = batch.xc_off_grid[0].cpu()
        yc_off_grid = (batch.yc_off_grid[0].cpu() * y_std_off) + y_mean_off

        xt = batch.xt[0].cpu()
        yt = (batch.yt[0].cpu() * y_std_off) + y_mean_off
        pred_mean_t = (pred_mean_t[0] * y_std_off) + y_mean_off
        pred_std_t = pred_std_t[0] * y_std_off

        x_grid = flatten_grid(x_grid)[0].cpu()
        y_grid = flatten_grid(y_grid)[0].cpu() * y_std_on + y_mean_on
        pred_mean_grid = (pred_mean_grid[0] * y_std_on) + y_mean_on
        pred_std_grid = pred_std_grid[0] * y_std_on

        diff_grid = y_grid - pred_mean_grid
        diff_grid_norm = diff_grid / pred_std_grid

        vmin, vmax = y_grid.min(), y_grid.max()
        scatter_kwargs = {
            "s": 15,
            "marker": "o",
            "alpha": 1.0,
            "vmin": vmin,
            "vmax": vmax,
            "lw": 0,
        }
        grid_args = scatter_kwargs | {"s": 2 if len(x_grid) < 50000 else 0.3, "marker": "s"}
        divnorm=colors.TwoSlopeNorm(vmin=diff_grid.min(), vcenter=0, vmax=diff_grid.max())
        diff_args = grid_args | {"norm":divnorm, "vmin": None, "vmax": None, "cmap": "seismic"}

        if subplots:
            fig, axes = plt.subplots(
                figsize=figsize,
                ncols=3,
                nrows=2,
                constrained_layout=True,
                subplot_kw={"projection": ccrs.PlateCarree()},
            )

            for ax in axes.flat:
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                # ax.gridlines()
                ax.set_axisbelow(True)


            axes[0, 0].scatter(xt[:, -1], xt[:, -2], c=pred_mean_t, **scatter_kwargs)
            axes[0, 0].set_title("Target predictions", fontsize=18)

            axes[0, 1].scatter(xt[:, -1], xt[:, -2], c=yt, **scatter_kwargs)
            axes[0, 1].set_title("Target true values", fontsize=18)

            row1 = axes[0, 2].scatter(xc_off_grid[:, -1], xc_off_grid[:, -2], c=yc_off_grid, **scatter_kwargs | {"s": 5})
            axes[0, 2].set_title("Off grid context", fontsize=18)

            row2_col1 = axes[1, 0].scatter(x_grid[:, -1], x_grid[:, -2], c=pred_mean_grid, **grid_args)
            axes[1, 0].set_title("Global predictions", fontsize=18)

            row2_col2 = axes[1, 1].scatter(x_grid[:, -1], x_grid[:, -2], c=y_grid, **grid_args)
            axes[1, 1].set_title("Global true values", fontsize=18)

            row2_diffs = axes[1, 2].scatter(x_grid[:, -1], x_grid[:, -2], c=diff_grid, **diff_args)
            axes[1, 2].set_title("Global difference", fontsize=18)

            # Add colourbar.
            fig.colorbar(row1, ax=axes[0], fraction=0.046, pad=0.012)
            fig.colorbar(row2_col1,ax=axes[1,0], fraction=0.046, pad=0.04)
            fig.colorbar(row2_col2,ax=axes[1,1], fraction=0.046, pad=0.04)
            fig.colorbar(row2_diffs, ax=axes[1, 2], fraction=0.046, pad=0.04)

            fname = f"fig/{name}/{i:03d}"
            if wandb.run is not None and logging:
                wandb.log({fname: wandb.Image(fig)})
            elif savefig:
                if not os.path.isdir(f"fig/{name}/{i:03d}"):
                    os.makedirs(f"fig/{name}/{i:03d}")
                plt.savefig(fname)
            else:
                plt.show()

            plt.close()

        else:
            grid_args["s"] = figsize[1] + 1.5
            diff_args["s"] = figsize[1] + 1.5
            std_args = grid_args | {"vmin": pred_std_grid.min(), "vmax": pred_std_grid.max()}

            for fig_name, x_plot, y_plot, sargs, in zip(
                ("Target predictions", "Target true values", "Off grid context", "Global predicted average", "Global true values", "Global error", "Global predicted uncertainty", "Normalised global error"),
                (xt, xt, xc_off_grid, x_grid, x_grid, x_grid, x_grid, x_grid),
                (pred_mean_t, yt, yc_off_grid, pred_mean_grid, y_grid, diff_grid, pred_std_grid, diff_grid_norm),
                (scatter_kwargs, scatter_kwargs, scatter_kwargs | {"s": 5}, grid_args, grid_args, diff_args, std_args, diff_args),
            ):
                fig = plt.figure(figsize=figsize, dpi=100)

                ax = plt.axes(projection=ccrs.PlateCarree())
                #ax.set_title(fig_name, fontsize=20)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                ax.set_axisbelow(True)

                gl = ax.gridlines(draw_labels=True)
                gl.xlabel_style = {"size": 15}
                gl.ylabel_style = {"size": 15}
                # ax.tick_params(axis="both", which="major", labelsize=20)

                sc = ax.scatter(
                        x_plot[:, -1], x_plot[:, -2], c=y_plot, **sargs
                    )

                # Add colourbar.
                if colorbar:
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.07)
                    cbar.ax.set_title("K")
                    cbar.solids.set(alpha=1)

                plt.tight_layout()

                fname = f"fig/{name}/{i:03d}/{fig_name}"
                if wandb.run is not None and logging:
                    wandb.log({fname: wandb.Image(fig)})
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}"):
                        os.makedirs(f"fig/{name}/{i:03d}")
                    plt.savefig(fname)
                else:
                    plt.show()

                plt.close()
