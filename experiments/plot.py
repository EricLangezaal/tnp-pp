import copy
import os
from typing import Callable, List, Tuple, Union

import matplotlib
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
    dim = batches[0].xc.shape[-1]

    if dim == 1:
        x_plot = torch.linspace(x_range[0], x_range[1], points_per_dim).to(
            batches[0].xc
        )[None, :, None]
        for i in range(num_fig):
            batch = batches[i]
            xc = batch.xc[:1]
            yc = batch.yc[:1]
            xt = batch.xt[:1]
            yt = batch.yt[:1]

            if isinstance(batch, OOTGBatch):
                batch.xc_on_grid = batch.xc_on_grid[:1]
                batch.xc_off_grid = batch.xc_off_grid[:1]
                batch.yc_on_grid = batch.yc_on_grid[:1]
                batch.yc_off_grid = batch.yc_off_grid[:1]

            batch.xc = xc
            batch.yc = yc
            batch.xt = xt
            batch.yt = yt

            plot_batch = copy.deepcopy(batch)
            plot_batch.xt = x_plot

            with torch.no_grad():
                y_plot_pred_dist = pred_fn(model, plot_batch)
                yt_pred_dist = pred_fn(model, batch)

            model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[..., 0].numel()
            mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

            # Make figure for plotting
            fig = plt.figure(figsize=figsize)

            # Plot context and target points
            plt.scatter(
                xc[0, :, 0].cpu().numpy(),
                yc[0, :, 0].cpu().numpy(),
                c="k",
                label="Context",
                s=30,
            )

            if plot_target:
                plt.scatter(
                    xt[0, :, 0].cpu().numpy(),
                    yt[0, :, 0].cpu().numpy(),
                    c="r",
                    label="Target",
                    s=30,
                )

            # Plot model predictions
            plt.plot(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu(),
                c="tab:blue",
                lw=3,
            )

            plt.fill_between(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
                mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            title_str = f"$N = {xc.shape[1]}$ NLL = {model_nll:.3f}"

            gt_mean, gt_std = None, None
            if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
                with torch.no_grad():
                    gt_mean, gt_std, _ = batch.gt_pred(
                        xc=xc[:1],
                        yc=yc[:1],
                        xt=x_plot[:1],
                        batch=batch
                    )
                    _, _, gt_loglik = batch.gt_pred(
                        xc=xc[:1],
                        yc=yc[:1],
                        xt=xt[:1],
                        yt=yt[:1],
                        batch=batch
                    )
                    gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

                # Plot ground truth
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu(),
                    "--",
                    color="tab:purple",
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

            plt.legend(loc="upper right", fontsize=20)
            plt.tight_layout()

            fname = f"fig/{name}/{i:03d}"
            if wandb.run is not None and logging:
                wandb.log({fname: wandb.Image(fig)})
            elif savefig:
                if not os.path.isdir(f"fig/{name}"):
                    os.makedirs(f"fig/{name}")
                plt.savefig(fname, bbox_inches="tight")
            else:
                plt.show()
                
    else:
        raise NotImplementedError
    
    plt.close()
