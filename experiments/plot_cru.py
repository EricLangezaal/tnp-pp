import copy
import os
from typing import Callable, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.base import Batch
from icicl.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_cru(
    model: Union[
        nn.Module,
        Callable[..., torch.distributions.Distribution],
    ],
    batches: List[Batch],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    time_idx: Optional[List[int]] = None,
    name: str = "plot",
    subplots: bool = True,
    savefig: bool = False,
    logging: bool = True,
    colorbar: bool = False,
    pred_fn: Callable = np_pred_fn,
    num_np_samples: int = 16,
):
    time_idx = [0, -1] if time_idx is None else time_idx

    for i in range(num_fig):
        batch = batches[i]
        xc = batch.xc[:1]
        yc = batch.yc[:1]
        xt = batch.xt[:1]
        yt = batch.yt[:1]
        x = batch.x[:1]
        y = batch.y[:1]

        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = yt
        batch.x = x
        batch.y = y

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = x

        if not isinstance(model, nn.Module):
            y_pred_dist = model(xc=xc, yc=yc, xt=x)
            yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

            # Detach gradients.
            y_pred_dist.loc = y_pred_dist.loc.detach().unsqueeze(0)
            y_pred_dist.scale = y_pred_dist.scale.detach().unsqueeze(0)
            yt_pred_dist.loc = yt_pred_dist.loc.detach().unsqueeze(0)
            yt_pred_dist.scale = yt_pred_dist.scale.detach().unsqueeze(0)
        else:
            with torch.no_grad():
                y_pred_dist = pred_fn(model, plot_batch, num_samples=num_np_samples)
                yt_pred_dist = pred_fn(model, batch, num_samples=num_np_samples)

        model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[:-1].numel()
        pred_mean, pred_std = y_pred_dist.mean.cpu(), y_pred_dist.stddev.cpu()

        # Rescale inputs and outputs.
        xc = (xc[..., :3].cpu() * x_std) + x_mean
        x = (x[..., :3].cpu() * x_std) + x_mean
        yc = (yc.cpu() * y_std) + y_mean
        y = (y.cpu() * y_std) + y_mean
        pred_mean = (pred_mean * y_std) + y_mean
        pred_std = pred_std * y_std

        # Get indicies corresponding to single time points.
        ts = x[0, :, 0].unique()
        for idx in time_idx:
            t = ts[idx]
            data_idx_c = torch.where(xc[0, :, 0] == t)[0]
            data_idx = torch.where(x[0, :, 0] == t)[0]

            # Get data for this time point.
            xc_ = xc[0, data_idx_c].cpu()
            yc_ = yc[0, data_idx_c, 0].cpu()
            x_ = x[0, data_idx].cpu()
            y_ = y[0, data_idx, 0].cpu()
            pred_mean_ = pred_mean[0, data_idx, 0]
            vmin = min(y_.min(), y_.max())
            vmax = max(y_.max(), y_.max())
            pred_std_ = pred_std[0, data_idx, 0]

            # Note that x is (time, lat, lon).
            ylim = (
                (x_[:, 1].min().item() - 0.5, x_[:, 1].max().item() + 0.5)
                if lat_range is None
                else lat_range
            )
            xlim = (
                (x_[:, 2].min().item() - 0.5, x_[:, 2].max().item() + 0.5)
                if lon_range is None
                else lon_range
            )
            scatter_kwargs = {
                "s": 15,
                "marker": "s",
                "alpha": 1.0,
                "vmin": vmin,
                "vmax": vmax,
                "lw": 0,
            }

            if subplots:
                fig, axes = plt.subplots(
                    figsize=figsize,
                    ncols=3,
                    nrows=1,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )

                for ax in axes:
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS)
                    # ax.gridlines()
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_axisbelow(True)

                axes[0].scatter(
                    xc_[:, 2],
                    xc_[:, 1],
                    c=yc_,
                    **scatter_kwargs,
                )
                axes[1].scatter(
                    x_[:, 2],
                    x_[:, 1],
                    c=pred_mean_,
                    **scatter_kwargs,
                )
                sc = axes[2].scatter(x_[:, 2], x_[:, 1], c=y_, **scatter_kwargs)

                axes[0].set_title("Context set", fontsize=18)
                axes[1].set_title("Predictive mean", fontsize=18)
                axes[2].set_title("True values", fontsize=18)

                # Add colourbar.
                cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
                cbar.solids.set(alpha=1)

                plt.suptitle(
                    f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}    "
                    #
                    f"NLL = {model_nll:.3f}",
                    fontsize=18,
                )

                fname = f"fig/{name}/{i:03d}/t-{t}"
                if wandb.run is not None and logging:
                    wandb.log({fname: wandb.Image(fig)})
                elif savefig:
                    if not os.path.isdir(f"fig/{name}/{i:03d}"):
                        os.makedirs(f"fig/{name}/{i:03d}")
                    plt.savefig(fname)
                else:
                    plt.show()

                # plt.tight_layout()
                plt.close()

            else:
                scatter_kwargs["s"] = 500
                for fig_name, x_plot, y_plot in zip(
                    ("context", "ground_truth", "pred_mean", "pred_std"),
                    (xc_, x_, x_, x_),
                    (yc_, y_, pred_mean_, pred_std_),
                ):
                    fig = plt.figure(figsize=figsize)

                    ax = plt.axes(projection=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_axisbelow(True)

                    gl = ax.gridlines(draw_labels=True)
                    gl.xlabel_style = {"size": 15}
                    gl.ylabel_style = {"size": 15}
                    # ax.tick_params(axis="both", which="major", labelsize=20)

                    if fig_name == "pred_std":
                        std_scatter_kwargs = scatter_kwargs
                        std_scatter_kwargs["vmin"] = y_plot.min()
                        std_scatter_kwargs["vmax"] = y_plot.max()
                        sc = ax.scatter(
                            x_plot[:, 2], x_plot[:, 1], c=y_plot, **std_scatter_kwargs
                        )
                    else:
                        sc = ax.scatter(
                            x_plot[:, 2], x_plot[:, 1], c=y_plot, **scatter_kwargs
                        )

                    # Add colourbar.
                    if colorbar:
                        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.09)
                        cbar.solids.set(alpha=1)

                    plt.tight_layout()

                    fname = f"fig/{name}/{i:03d}/t-{t}/{fig_name}"
                    if wandb.run is not None and logging:
                        wandb.log({fname: wandb.Image(fig)})
                    elif savefig:
                        if not os.path.isdir(f"fig/{name}/{i:03d}/t-{t}"):
                            os.makedirs(f"fig/{name}/{i:03d}/t-{t}")
                        plt.savefig(fname)
                    else:
                        plt.show()

                    plt.close()
