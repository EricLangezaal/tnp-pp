from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from utils import ar_predict

import wandb
from icicl.data.base import Batch, ICBatch
from icicl.data.synthetic import SyntheticBatch
from icicl.models.telbanp import TELBANP

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot(
    model: nn.Module,
    batches: List[Batch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (-5.0, 5.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 256,
    plot_ar_mode: bool = False,
    num_ar_samples: int = 20,
    name: str = "plot",
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

            with torch.no_grad():
                if isinstance(batch, ICBatch):
                    y_plot_pred_dist = model(
                        xc=xc,
                        yc=yc,
                        xic=batch.xic[:1],
                        yic=batch.yic[:1],
                        xt=x_plot,
                    )
                    yt_pred_dist = model(
                        xc=xc, yc=yc, xic=batch.xic[:1], yic=batch.yic[:1], xt=xt
                    )
                else:
                    assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
                    y_plot_pred_dist = model(xc=xc, yc=yc, xt=x_plot)
                    yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

                model_nll = -yt_pred_dist.log_prob(yt).mean()
                mean, std = y_plot_pred_dist.loc, y_plot_pred_dist.scale

            # Make figure for plotting
            fig = plt.figure(figsize=figsize)

            # Plot context and target points
            plt.scatter(
                xc[0, :, 0].numpy(),
                yc[0, :, 0].numpy(),
                c="k",
                label="Context",
                s=20,
            )

            plt.scatter(
                xt[0, :, 0].numpy(),
                yt[0, :, 0].numpy(),
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

            title_str = f"$N = {xc.shape[1]}$ NLL = {model_nll:.3f}"

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

                title_str += f" GT NLL = {gt_nll:.3f}"

            if plot_ar_mode and not isinstance(batch, ICBatch):
                ar_x_plot = torch.linspace(xc.max(), x_range[1], 50).to(batches[0].xc)[
                    None, :, None
                ]
                _, ar_sample_logprobs = ar_predict(
                    model, xc=xc, yc=yc, xt=xt, num_samples=num_ar_samples
                )
                ar_samples, _ = ar_predict(
                    model, xc=xc, yc=yc, xt=ar_x_plot, num_samples=num_ar_samples
                )

                for ar_sample in ar_samples[:1]:
                    plt.plot(
                        ar_x_plot[0, :, 0].cpu(),
                        ar_sample[0, :, 0].cpu(),
                        color="tab:blue",
                        alpha=0.4,
                        label="AR samples",
                    )

                ar_nll = -ar_sample_logprobs.mean()
                title_str += f" AR NLL: {ar_nll:.3f}"

            if isinstance(model, TELBANP):
                xq = model.encoder.nested_perceiver_encoder.tq_cache

                if xq is not None:
                    plt.scatter(
                        xq[0, ...],
                        torch.ones_like(xq[0, ...])
                        * (y_lim[0] + 0.05 * (y_lim[-1] - y_lim[0])),
                        color="tab:red",
                        label="Pseudo-locations",
                        s=20,
                    )

            plt.title(title_str, fontsize=24)

            # Set axis limits
            plt.xlim(x_range)
            plt.ylim(y_lim)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.legend(loc="upper right", fontsize=14)
            if wandb.run is not None:
                wandb.log({f"fig/{name}/{i:03d}": wandb.Image(fig)})
            else:
                plt.show()

            plt.close()

    elif dim in (2, 3):
        # Plots first two dimensions in the case of 3.
        figsize = (24.0, 8.0)
        for i in range(num_fig):
            batch = batches[i]
            x = batch.x[:1]
            y = batch.y[:1]
            xc = batch.xc[:1]
            yc = batch.yc[:1]
            xt = batch.xt[:1]
            yt = batch.yt[:1]

            with torch.no_grad():
                if isinstance(batch, ICBatch):
                    y_plot_pred_dist = model(
                        xc=xc,
                        yc=yc,
                        xic=batch.xic,
                        yic=batch.yic,
                        xt=x,
                    )
                    yt_pred_dist = model(
                        xc=xc, yc=yc, xic=batch.xic[:1], yic=batch.yic[:1], xt=xt
                    )
                else:
                    assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
                    y_plot_pred_dist = model(xc=xc, yc=yc, xt=x)
                    yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

                model_nll = -yt_pred_dist.log_prob(yt).mean()
                mean, std = y_plot_pred_dist.loc, y_plot_pred_dist.scale

            # Get indices corresponding to single time point.
            for t_min in x[0, :, 0].unique():
                # Make figure for plotting
                fig, axes = plt.subplots(
                    figsize=figsize, ncols=3, nrows=1, sharex=True, sharey=True
                )

                t_min_idx_c = torch.where(xc[0, :, 0] == t_min)[0]
                t_min_idx = torch.where(x[0, :, 0] == t_min)[0]

                axes[0].scatter(
                    xc[0, t_min_idx_c, 1],
                    xc[0, t_min_idx_c, 2],
                    c=yc[0, t_min_idx_c, 0],
                    s=50,
                )
                axes[1].scatter(
                    x[0, t_min_idx, 1],
                    x[0, t_min_idx, 2],
                    c=mean[0, t_min_idx, 0],
                    s=50,
                )
                axes[2].scatter(
                    x[0, t_min_idx, 1], x[0, t_min_idx, 2], c=y[0, t_min_idx, 0], s=50
                )

                axes[0].set_title("Context set", fontsize=18)
                axes[1].set_title("Predictive mean", fontsize=18)
                axes[2].set_title("True values", fontsize=18)

                plt.suptitle(
                    f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}    "
                    #
                    f"NLL = {model_nll:.3f}",
                    fontsize=24,
                )

                if wandb.run is not None:
                    wandb.log({f"fig/{name}/{i:03d}/t-{t_min}": wandb.Image(fig)})
                else:
                    plt.show()

                plt.close()
    else:
        raise NotImplementedError
