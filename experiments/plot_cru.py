from typing import List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.base import Batch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_cru(
    model: nn.Module,
    batches: List[Batch],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    lat_range: Tuple[float, float] = (35.25, 59.75),
    lon_range: Tuple[float, float] = (10.25, 44.75),
    time_idx: Optional[List[int]] = None,
    name: str = "plot",
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

        with torch.no_grad():
            y_pred_dist = model(xc=xc, yc=yc, xt=x)
            yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

            model_nll = -yt_pred_dist.log_prob(yt).mean().cpu()
            pred_mean, pred_std = y_pred_dist.loc.cpu(), y_pred_dist.scale.cpu()

        # Rescale inputs and outputs.
        xc = (xc.cpu() * x_std) + x_mean
        x = (x.cpu() * x_std) + x_mean
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
            # pred_std_ = pred_std[data_idx]

            # Note that x is (time, lat, lon).

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
                ax.set_xlim([lon_range[0], lon_range[1]])
                ax.set_ylim([lat_range[0], lat_range[1]])
                ax.set_axisbelow(True)

            vmin = min(pred_mean_.min(), y_.max())
            vmax = max(pred_mean_.max(), y_.max())

            # # Calculate marker size.
            # num_x_points = (lon_range[1] - lon_range[0]) // 0.5
            # num_y_points = (lat_range[1] - lat_range[0]) // 0.5

            # # Distance is 1 / num_x_points units.
            # bbox = (
            #     axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # )
            # width = bbox.width * fig.dpi
            # height = bbox.height * fig.dpi

            # mw = width / num_x_points
            # mh = height / num_y_points
            # ms = (mh * mw) ** 0.5

            ms = 15
            axes[0].scatter(
                xc_[:, 2],
                xc_[:, 1],
                c=yc_,
                alpha=1.0,
                marker="s",
                # s=25,
                s=ms,
                vmin=vmin,
                vmax=vmax,
                lw=0,
            )
            axes[1].scatter(
                x_[:, 2],
                x_[:, 1],
                c=pred_mean_,
                alpha=1.0,
                marker="s",
                s=ms,
                vmin=vmin,
                vmax=vmax,
                lw=0,
            )
            sc = axes[2].scatter(
                x_[:, 2],
                x_[:, 1],
                c=y_,
                alpha=1.0,
                marker="s",
                s=ms,
                vmin=vmin,
                vmax=vmax,
                lw=0,
            )

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

            if wandb.run is not None:
                wandb.log({f"fig/{name}/{i:03d}/t-{t}": wandb.Image(fig)})
            else:
                plt.show()

            # plt.tight_layout()
            plt.close()
