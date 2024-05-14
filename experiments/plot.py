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
from icicl.models.telbanp import TELBANP
from icicl.utils.experiment_utils import ar_predict, np_pred_fn

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
    pred_fn: Callable = np_pred_fn,
    num_np_samples: int = 16,
    plot_te_part: bool = False,
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
                    y_plot_pred_dist = pred_fn(
                        model, plot_batch, num_samples=num_np_samples
                    )
                    yt_pred_dist = pred_fn(model, batch, num_samples=num_np_samples)

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

            if isinstance(y_plot_pred_dist, torch.distributions.MixtureSameFamily):
                # Plot individual component means.
                for i in range(num_np_samples):
                    # Get component distribution.
                    sample_mean = y_plot_pred_dist.component_distribution.mean[
                        :, i, ...
                    ]
                    plt.plot(
                        x_plot[0, :, 0].cpu(),
                        sample_mean[0, :, 0].cpu(),
                        c="tab:blue",
                        lw=1,
                        alpha=0.5,
                    )

            title_str = f"$N = {xc.shape[1]}$ NLL = {model_nll:.3f}"

            if plot_te_part and isinstance(model, TELBANP):
                model_copy = copy.deepcopy(model)
                model_copy.encoder.nested_perceiver_encoder.p_basis_dropout = 1.0
                model_copy.train()

                with torch.no_grad():
                    te_y_plot_pred_dist = pred_fn(
                        model_copy, plot_batch, num_samples=num_np_samples
                    )

                te_mean, te_std = te_y_plot_pred_dist.mean, te_y_plot_pred_dist.stddev
                # Plot model predictions
                plt.plot(
                    x_plot[0, :, 0].cpu(),
                    te_mean[0, :, 0].cpu(),
                    c="tab:red",
                    lw=3,
                )

                plt.fill_between(
                    x_plot[0, :, 0].cpu(),
                    te_mean[0, :, 0].cpu() - 2.0 * te_std[0, :, 0].cpu(),
                    te_mean[0, :, 0].cpu() + 2.0 * te_std[0, :, 0].cpu(),
                    color="tab:red",
                    alpha=0.2,
                    label="TE Model",
                )

            gt_mean, gt_std = None, None
            if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
                #if isinstance(batch, OOTGBatch):
                #    # restore 2d output temporarily
                #    num_oftg = batch.xc_off_grid.shape[-2] + batch.xt.shape[-2]
                #    yc = torch.cat((batch.y[:, :batch.xc_off_grid.shape[-2], :], batch.y[:, num_oftg:, :]), dim=-2)
                #    yt = batch.y[:, batch.xc_off_grid.shape[-2]: num_oftg, :]
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

            plt.title(title_str, fontsize=24)
            plt.grid()

            # Set axis limits
            plt.xlim(x_range)
            # plt.ylim(y_lim)
            if gt_mean is not None and gt_std is not None:
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
