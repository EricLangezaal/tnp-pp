from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.data import SyntheticBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: nn.Module,
    batches: List[SyntheticBatch],
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
            xc = batches[i].xc[:1]
            yc = batches[i].yc[:1]
            xt = batches[i].xt[:1]
            yt = batches[i].yt[:1]

            if hasattr(batches[i], "xic") and hasattr(batches[i], "yic"):
                xic = batches[i].xic[:1]
                yic = batches[i].yic[:1]
            else:
                xic = None
                yic = None

            gt_pred = batches[i].gt_pred

            with torch.no_grad():
                if xic is not None and yic is not None:
                    y_plot_pred_dist = model(
                        xc=xc, yc=yc, xic=xic, yic=yic, xt=x_plot[:1]
                    )
                    yt_pred_dist = model(xc=xc, yc=yc, xic=xic, yic=yic, xt=xt)
                else:
                    y_plot_pred_dist = model(xc=xc, yc=yc, xt=x_plot[:1])
                    yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

                mean, std = y_plot_pred_dist.loc, y_plot_pred_dist.scale
                gt_mean, gt_std, _ = gt_pred(
                    xc=xc,
                    yc=yc,
                    xt=x_plot[:1],
                )

                model_nll = -yt_pred_dist.log_prob(yt).mean()
                _, _, gt_loglik = gt_pred(
                    xc=xc,
                    yc=yc,
                    xt=xt,
                    yt=yt,
                )
                gt_nll = -gt_loglik.mean() / batches[i].yt.shape[1]

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
                x_plot[0, :, 0],
                mean[0, :, 0],
                c="tab:blue",
            )

            plt.fill_between(
                x_plot[0, :, 0],
                mean[0, :, 0] - 2.0 * std[0, :, 0],
                mean[0, :, 0] + 2.0 * std[0, :, 0],
                color="tab:blue",
                alpha=0.2,
                label="Model",
            )

            # Plot ground truth
            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :],
                "--",
                color="tab:purple",
            )

            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :] + 2 * gt_std[0, :],
                "--",
                color="tab:purple",
            )

            plt.plot(
                x_plot[0, :, 0],
                gt_mean[0, :] - 2 * gt_std[0, :],
                "--",
                color="tab:purple",
                label="Ground truth",
            )

            # Set axis limits
            plt.xlim(x_range)
            plt.ylim(y_lim)

            # Set title
            nc = xc.shape[1]
            # lengthscale = gt_pred.kernel.lengthscale.detach().item()
            plt.title(
                f"$N = {nc}$   "
                # f"$\\ell$ = {lengthscale:.2f}  "
                f"NLL = {model_nll:.3f} \t" f"GT NLL = {gt_nll:.3f}",
                fontsize=24,
            )

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            wandb.log({f"fig/epoch-{epoch:04d}/{i:03d}": wandb.Image(fig)})
            plt.close()

    else:
        raise NotImplementedError
