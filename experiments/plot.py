from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.data import Batch, ICBatch, SyntheticBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: nn.Module,
    batches: List[Batch],
    epoch: int = 0,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 512,
):
    # Get dimension of input data
    dim = batches[0].xc.shape[-1]

    if dim == 1:
        x_plot = torch.linspace(x_range[0], x_range[1], points_per_dim)[None, :, None]
        for i in range(num_fig):
            batch = batches[i]
            xc = batch.xc
            yc = batch.yc
            xt = batch.xt
            yt = batch.yt

            with torch.no_grad():
                if isinstance(batch, ICBatch):
                    y_plot_pred_dist = model(
                        xc=xc,
                        yc=yc,
                        xic=batch.xic,
                        yic=batch.yic,
                        xt=x_plot[:1].repeat(xc.shape[0], 1, 1),
                    )
                    yt_pred_dist = model(
                        xc=xc, yc=yc, xic=batch.xic, yic=batch.yic, xt=xt
                    )
                else:
                    assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
                    y_plot_pred_dist = model(
                        xc=xc, yc=yc, xt=x_plot[:1].repeat(xc.shape[0], 1, 1)
                    )
                    yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

                model_nll = -yt_pred_dist.log_prob(yt)[:1].mean()
                mean, std = y_plot_pred_dist.loc[:1], y_plot_pred_dist.scale[:1]

            # Make figure for plotting
            fig = plt.figure(figsize=figsize)

            # Plot context and target points
            plt.scatter(
                xc[0, :, 0],
                yc[0, :, 0],
                c="k",
                label="Context",
                s=20,
            )

            plt.scatter(
                xt[0, :, 0],
                yt[0, :, 0],
                c="r",
                label="Target",
                s=20,
            )

            # Plot model predictions
            plt.plot(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu(),
                c="tab:blue",
            )

            plt.fill_between(
                x_plot[0, :, 0].cpu(),
                mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
                mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
                with torch.no_grad():
                    gt_mean, gt_std, _ = batch.gt_pred(
                        xc=xc[:1],
                        yc=yc[:1],
                        xt=x_plot[:1],
                    )
                    _, _, gt_loglik = batch.gt_pred(
                        xc=xc[:1],
                        yc=yc[:1],
                        xt=xt[:1],
                        yt=yt[:1],
                    )
                    gt_nll = -gt_loglik.mean() / yt.shape[1]

                # Plot ground truth
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu(),
                    "--",
                    color="tab:purple",
                )

                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                )

                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
                    "--",
                    color="tab:purple",
                    label="Ground truth",
                )

                plt.title(
                    f"$N = {xc.shape[1]}$   "
                    # f"$\\ell$ = {lengthscale:.2f}  "
                    f"NLL = {model_nll:.3f} \t" f"GT NLL = {gt_nll:.3f}",
                    fontsize=24,
                )
            else:
                plt.title(
                    f"$N = {xc.shape[1]}$   "
                    # f"$\\ell$ = {lengthscale:.2f}  "
                    f"NLL = {model_nll:.3f} \t",
                    fontsize=24,
                )

            # Set axis limits
            plt.xlim(x_range)
            plt.ylim(y_lim)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            wandb.log({f"fig/epoch-{epoch:04d}/{i:03d}": wandb.Image(fig)})
            plt.close()

    else:
        raise NotImplementedError
