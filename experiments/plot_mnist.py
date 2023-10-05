from typing import List, Tuple

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import wandb
from icicl.data.mnist import MNISTBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_mnist(
    model: nn.Module,
    batches: List[MNISTBatch],
    epoch: int = 0,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (24.0, 8.0),
):
    for i in range(num_fig):
        x = batches[i].x
        y = batches[i].y
        xc = batches[i].xc
        yc = batches[i].yc
        xt = batches[i].xt
        yt = batches[i].yt
        mc = batches[i].mc

        if hasattr(batches[i], "xic") and hasattr(batches[i], "yic"):
            xic = batches[i].xic
            yic = batches[i].yic
        else:
            xic = None
            yic = None

        with torch.no_grad():
            if xic is not None and yic is not None:
                y_plot_pred_dist = model(
                    xc=xc,
                    yc=yc,
                    xic=xic,
                    yic=yic,
                    xt=x,
                )
                yt_pred_dist = model(xc=xc, yc=yc, xic=xic, yic=yic, xt=xt)
            else:
                y_plot_pred_dist = model(xc=xc, yc=yc, xt=x)
                yt_pred_dist = model(xc=xc, yc=yc, xt=xt)

            mean, std = (
                y_plot_pred_dist.loc[:1].numpy(),
                y_plot_pred_dist.scale[:1].numpy(),
            )
            model_nll = -yt_pred_dist.log_prob(yt)[:1].mean()

        # Make figure for plotting
        fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=1)

        # Reorganise into grid.
        y_plot = np.ma.masked_where(
            ~mc[:1, :],
            y[:1, :, 0].numpy(),
        )
        y_plot = einops.rearrange(y_plot, "1 (n m) -> n m", n=28, m=28)
        mean = einops.rearrange(mean[..., 0], "1 (n m) -> n m", n=28, m=28)
        std = einops.rearrange(std[..., 0], "1 (n m) -> n m", n=28, m=28)

        axes[0].imshow(y_plot)
        axes[1].imshow(mean)
        axes[2].imshow(std)

        axes[0].set_title("Context set", fontsize=18)
        axes[1].set_title("Mean prediction", fontsize=18)
        axes[2].set_title("Std prediction", fontsize=18)

        plt.suptitle(
            f"prop_ctx = {xc.shape[-2] / x.shape[-2]:.2f}    "
            #
            f"NLL = {model_nll:.3f}",
            fontsize=24,
        )

        wandb.log({f"fig/epoch-{epoch:04d}/{i:03d}": wandb.Image(fig)})
        plt.close()
