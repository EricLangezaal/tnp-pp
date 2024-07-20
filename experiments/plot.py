import copy
import os
from typing import Callable, List, Tuple, Union

import warnings
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.base import Batch
from icicl.data.on_off_grid import OOTGBatch
from icicl.data.synthetic import SyntheticBatch
from icicl.utils.experiment_utils import np_pred_fn

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: Union[
        nn.Module,
        Callable[..., torch.distributions.Distribution],
    ],
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (-5.0, 5.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 256,
    plot_target: bool = True,
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
):
    # Get dimension of input data
    dim = batches[0].xt.shape[-1]
    if dim not in [1,2]:
        warnings.warn("Plotting only supported for 1D and 2D data.")
        return
    
    if dim == 1:
        x_plot = torch.linspace(x_range[0], x_range[1], points_per_dim).to(
            batches[0].xt
        )[None, :, None]
    else:
        x_plot = torch.stack(
            torch.meshgrid(
                *[torch.linspace(x_range[0], x_range[1], int(points_per_dim ** (1/dim))) for _ in range(dim)],
                indexing='ij'
            ),
            axis=-1,
        ).view(-1, dim).unsqueeze(0).to(batches[0].xt)

    for i in range(num_fig):
        batch = batches[i]

        for key, value in vars(batch).items():
            if isinstance(value, torch.Tensor):
                setattr(batch, key, value[:1])

        plot_batch = copy.deepcopy(batch)
        plot_batch.xt = x_plot

        xc_plot, yc_plot = batch.xc, batch.yc
        if isinstance(batch, OOTGBatch):
            xc_plot = batch.used_modality.get(batch.xc_on_grid, batch.xc_off_grid, batch.xc)
            yc_plot = batch.used_modality.get(batch.yc_on_grid, batch.yc_off_grid, batch.yc)

        with torch.no_grad():
            y_plot_pred_dist = pred_fn(model, plot_batch)
            yt_pred_dist = pred_fn(model, batch)

        model_nll = -yt_pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
        mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

        title_str = f"$N = {xc_plot.shape[1]}$ NLL = {model_nll:.3f}"

        if dim == 1:
            fig = plt.figure(figsize=figsize)
             # Plot context and target points
            plt.scatter(
                xc_plot[0, :, 0].cpu().numpy(),
                yc_plot[0, :, 0].cpu().numpy(),
                c="k",
                label="Context",
                s=30,
            )
            if plot_target:
                plt.scatter(
                    batch.xt[0, :, 0].cpu().numpy(),
                    batch.yt[0, :, 0].cpu().numpy(),
                    c="r",
                    label="Target",
                    s=30,
                )

            # Plot model predictions
            plt.plot(
                x_plot[0, :, 0].cpu().numpy(),
                mean[0, :, 0].cpu(),
                color="tab:blue",
                lw=3,
                label="Model",
            )
            plt.fill_between(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
                mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
            )

        else:
            fig = plt.figure(figsize=(figsize[0] + 2, figsize[1]))
            ax = fig.add_subplot(projection='3d')

            ax.scatter(
                *[t.cpu().numpy() for t in torch.unbind(xc_plot[0], dim=-1)],
                yc_plot[0, :, 0].cpu().numpy(),
                c="k",
                label="Context",
                s=30,
            )
            if plot_target:
                ax.scatter(
                    *[t.cpu().numpy() for t in torch.unbind(batch.xt[0], dim=-1)],
                    batch.yt[0, :, 0].cpu().numpy(),
                    c="r",
                    label="Target",
                    s=30,
                )
            # Plot model predictions
            ax.plot_trisurf(
                *[t.cpu().numpy() for t in torch.unbind(x_plot[0], dim=-1)],
                mean[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.3,
                lw=3,
                label="Model",
            )

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            with torch.no_grad():
                gt_mean, gt_std, _ = batch.gt_pred(
                    xc=batch.xc[:1],
                    yc=batch.yc[:1],
                    xt=x_plot[:1],
                    batch=batch
                )
                _, _, gt_loglik = batch.gt_pred(
                    xc=batch.xc[:1],
                    yc=batch.yc[:1],
                    xt=batch.xt[:1],
                    yt=batch.yt[:1],
                    batch=batch
                )
                gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

            if dim == 1:
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    label="Ground truth",
                    lw=3,
                )
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    lw=3,
                )
            else:
                ax.plot_trisurf(
                    *[t.cpu().numpy() for t in torch.unbind(x_plot[0], dim=-1)],
                    gt_mean[0, :].cpu(),
                    alpha=0.3,
                    color="tab:purple",
                    label="Ground truth",
                    lw=3,
                )
            
            title_str += f" GT NLL = {gt_nll:.3f}"

        plt.title(title_str, fontsize=24)
        plt.grid()

        # Set axis limits
        plt.xlim(x_range)
        plt.ylim(y_lim)

        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        if dim == 1:
            plt.tight_layout()
            plt.legend(loc="upper right", fontsize=20)
        else:
            ax.tick_params(labelsize=20)
            plt.legend(loc="upper right", fontsize=16, bbox_to_anchor=(1.5, 1))

        fname = f"fig/{name}/{i:03d}"
        if wandb.run is not None and logging:
            wandb.log({fname: wandb.Image(fig)})
        elif savefig:
            if not os.path.isdir(f"fig/{name}"):
                os.makedirs(f"fig/{name}")
            fig.savefig(fname)
        else:
            plt.show()
        plt.close()
