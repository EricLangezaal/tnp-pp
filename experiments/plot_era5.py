import copy
import os
from typing import Callable, List, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
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
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    colorbar: bool = False,
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = flatten_grid(plot_batch.xc_on_grid)

        with torch.no_grad():
            yt_pred_dist = pred_fn(model, batch)
            pred_mean_t, pred_std_t = yt_pred_dist.mean.cpu(), yt_pred_dist.stddev.cpu()
           
            yplot_pred_dist = pred_fn(model, plot_batch)
            pred_mean_grid, pred_std_grid = (
                yplot_pred_dist.mean.cpu(),
                yplot_pred_dist.stddev.cpu(),
            )

        # Rescale inputs and outputs.
        xc_off_grid = batch.xc_off_grid[0].cpu()
        yc_off_grid = (batch.yc_off_grid[0].cpu() * y_std) + y_mean

        xt = batch.xt[0].cpu()
        yt = (batch.yt[0].cpu() * y_std) + y_mean
        pred_mean_t = (pred_mean_t[0] * y_std) + y_mean
        pred_std_t = pred_std_t[0] * y_std

        xc_on_grid = flatten_grid(batch.xc_on_grid)[0].cpu()
        yc_on_grid = flatten_grid(batch.yc_on_grid)[0].cpu() * y_std + y_mean

        pred_mean_grid = (pred_mean_grid[0] * y_std) + y_mean
        pred_std_grid = pred_std_grid[0] * y_std

        diff_grid = yc_on_grid - pred_mean_grid
        diff_grid_norm = diff_grid / pred_std_grid

        vmin, vmax = yc_on_grid.min(), yc_on_grid.max()
        scatter_kwargs = {
            "s": 15,
            "marker": "s",
            "alpha": 1.0,
            "vmin": vmin,
            "vmax": vmax,
            "lw": 0,
        }
        diff_args = {"vmin": diff_grid.min(), "vmax": diff_grid.max(), "cmap": "seismic"}

        if subplots:
            fig, axes = plt.subplots(
                figsize=figsize,
                ncols=3,
                nrows=2,
                constrained_layout=True,
                subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
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

            row1 = axes[0, 2].scatter(xc_off_grid[:, -1], xc_off_grid[:, -2], c=yc_off_grid, **scatter_kwargs)
            axes[0, 2].set_title("Off grid context", fontsize=18)

            row2_col1 = axes[1, 0].scatter(xc_on_grid[:, -1], xc_on_grid[:, -2], c=pred_mean_grid, **scatter_kwargs)
            axes[1, 0].set_title("Global predictions", fontsize=18)

            row2_col2 = axes[1, 1].scatter(xc_on_grid[:, -1], xc_on_grid[:, -2], c=yc_on_grid, **scatter_kwargs)
            axes[1, 1].set_title("Global true values", fontsize=18)

            row2_diffs = axes[1, 2].scatter(xc_on_grid[:, -1], xc_on_grid[:, -2], c=diff_grid, **scatter_kwargs | diff_args)
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
            scatter_kwargs["s"] = 10
            for fig_name, x_plot, y_plot in zip(
                ("Target predictions", "Target true values", "Off grid context", "Global predictions", "Global true values", "Global difference", "Normalised global difference"),
                (xt, xt, xc_off_grid, xc_on_grid, xc_on_grid, xc_on_grid, xc_on_grid),
                (pred_mean_t, yt, yc_off_grid, pred_mean_grid, yc_on_grid, diff_grid, diff_grid_norm),
            ):
                fig = plt.figure(figsize=figsize)

                ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                ax.set_axisbelow(True)

                gl = ax.gridlines(draw_labels=True)
                gl.xlabel_style = {"size": 15}
                gl.ylabel_style = {"size": 15}
                # ax.tick_params(axis="both", which="major", labelsize=20)

                if "difference" in fig_name.lower():
                    diff_scatter_kwargs = scatter_kwargs
                    diff_scatter_kwargs["vmin"] = y_plot.min()
                    diff_scatter_kwargs["vmax"] = y_plot.max()
                    diff_scatter_kwargs["cmap"] = "seismic"
                    sc = ax.scatter(
                        x_plot[:, -1], x_plot[:, -2], c=y_plot, **diff_scatter_kwargs
                    )
                else:
                    sc = ax.scatter(
                        x_plot[:, -1], x_plot[:, -2], c=y_plot, **scatter_kwargs
                    )

                # Add colourbar.
                if colorbar:
                    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.09)
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