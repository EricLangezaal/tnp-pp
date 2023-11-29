from typing import List, Tuple

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import wandb
from icicl.data.image import ICImageBatch, ImageBatch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_image(
    model: nn.Module,
    batches: List[ImageBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (24.0, 8.0),
    name: str = "plot_image",
):
    for i in range(num_fig):
        batch = batches[i]

        x = batch.x
        y = batch.y
        xc = batch.xc
        yc = batch.yc
        xt = batch.xt
        yt = batch.yt
        mc = batch.mc

        with torch.no_grad():
            if isinstance(batch, ICImageBatch):
                y_plot_pred_dist = model(
                    xc=xc,
                    yc=yc,
                    xic=batch.xic,
                    yic=batch.yic,
                    xt=x,
                )
                yt_pred_dist = model(xc=xc, yc=yc, xic=batch.xic, yic=batch.yic, xt=xt)
            else:
                assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
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
        w = int(y_plot.shape[-1] ** 0.5)
        y_plot = einops.rearrange(y_plot, "1 (n m) -> n m", n=w, m=w)
        mean = einops.rearrange(mean[..., 0], "1 (n m) -> n m", n=w, m=w)
        std = einops.rearrange(std[..., 0], "1 (n m) -> n m", n=w, m=w)

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

        if wandb.run is not None:
            wandb.log({f"fig/{name}/{i:03d}": wandb.Image(fig)})
        else:
            plt.show()

        plt.close()
