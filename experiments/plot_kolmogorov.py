import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.base import Batch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_kolmogorov(
    model: nn.Module,
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (18.0, 5.0),
    plot_dims: Tuple[int, int] = (0, 1),
    other_dim_slice: int = 0,
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

        with torch.no_grad():
            y_pred_dist = model(xc=xc, yc=yc, xt=x)
            yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

        model_nll = -yt_pred_dist.log_prob(yt).mean()
        mean = y_pred_dist.loc

    fig = plt.figure(figsize=figsize)

    other_dim = [i for i in range(x.shape[-1]) if i not in plot_dims][0]

    # Select data to plot.
    other_dim_vals = x[0, :, other_dim].unique()
    other_dim_val = other_dim_vals[other_dim_slice]

    data_idx = torch.where(x[0, ..., other_dim] == other_dim_val)[0]
    data_idx_c = torch.where(xc[0, ..., other_dim] == other_dim_val)[0]

    x_ = x[0, data_idx]
    y_ = y[0, data_idx]
    xc_ = xc[0, data_idx_c]
    yc_ = yc[0, data_idx_c]
    mean_ = mean[0, data_idx]

    vmin = min(y_.min(), mean_.min())
    vmax = max(y_.max(), mean_.max())

    scatter_kwargs = {
        "s": 45,
        "marker": "s",
        "alpha": 1.0,
        "vmin": vmin,
        "vmax": vmax,
        "lw": 0,
    }

    xmin = x_[:, plot_dims[0]].min()
    xmax = x_[:, plot_dims[0]].max()
    ymin = x_[:, plot_dims[1]].min()
    ymax = x_[:, plot_dims[1]].max()
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)

    if subplots:
        fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=y.shape[-1])

        for y_dim in range(y.shape[-1]):
            for i, (x_plot, y_plot) in enumerate(zip((xc_, x_, x_), (yc_, y_, mean_))):
                if y.shape[-1] == 1:
                    ax = axes[i]
                else:
                    ax = axes[y_dim, i]

                sc = ax.scatter(
                    x_plot[:, plot_dims[0]],
                    x_plot[:, plot_dims[1]],
                    c=y_plot[:, y_dim],
                    **scatter_kwargs,
                )
                ax.set_aspect("equal")
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        axes.flatten()[0].set_title("Context set", fontsize=18)
        axes.flatten()[1].set_title("Ground truth", fontsize=18)
        axes.flatten()[2].set_title("Predictive mean", fontsize=18)

        # Add colourbar.
        cbar = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.04)
        cbar.solids.set(alpha=1)

        plt.suptitle(
            f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}    "
            #
            f"NLL = {model_nll:.3f}",
            fontsize=18,
        )

        fname = f"fig/{name}/{i:03d}"
        if wandb.run is not None and logging:
            wandb.log({fname: wandb.Image(fig)})
        elif savefig:
            if not os.path.isdir(f"fig/{name}"):
                os.makedirs(f"fig/{name}")
            plt.savefig(fname)
        else:
            plt.show()

    else:
        return
