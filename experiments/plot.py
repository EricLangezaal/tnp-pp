import os
from typing import Callable, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

import wandb
from icicl.data.base import Batch
from icicl.data.synthetic import SyntheticBatch
from icicl.models.telbanp import TELBANP
from icicl.utils.experiment_utils import ar_predict

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
    plot_ar_mode: bool = False,
    num_ar_samples: int = 20,
    plot_target: bool = True,
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
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

            if not isinstance(model, nn.Module):
                # model is a callable function.
                y_pred_dist = model(xc=xc, yc=yc, xt=x_plot)
                yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

                # Detach gradients.
                y_pred_dist.loc = y_pred_dist.loc.detach().unsqueeze(0)
                y_pred_dist.scale = y_pred_dist.scale.detach().unsqueeze(0)
                yt_pred_dist.loc = yt_pred_dist.loc.detach().unsqueeze(0)
                yt_pred_dist.scale = yt_pred_dist.scale.detach().unsqueeze(0)
            else:
                with torch.no_grad():
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
                s=30,
            )

            if plot_target:
                plt.scatter(
                    xt[0, :, 0].numpy(),
                    yt[0, :, 0].numpy(),
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

            if plot_ar_mode:
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
                        color="tab:green",
                        marker="^",
                        label="Pseudo-locations",
                        s=20,
                    )

            # plt.title(title_str, fontsize=24)
            plt.grid()

            # Set axis limits
            plt.xlim(x_range)
            # plt.ylim(y_lim)
            y_max = 0.25 + max(gt_mean[0, ...] + 2 * gt_std[0, ...])
            y_min = -0.25 + min(gt_mean[0, ...] - 2 * gt_std[0, ...])
            y_lim = (y_min, y_max)
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

            plt.close()

            plt.close()
    else:
        raise NotImplementedError
