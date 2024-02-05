import dataclasses
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from plot import plot
from plot_cru import plot_cru
from plot_image import plot_image
from plot_kolmogorov import plot_kolmogorov
from torch import nn
from utils import ModelCheckpointer, np_loss_fn

from icicl.data.base import Batch, ICBatch
from icicl.data.cru import CRUDataGenerator
from icicl.data.image import GriddedImageBatch, ImageGenerator
from icicl.data.kolmogorov import KolmogorovGenerator
from icicl.data.synthetic import SyntheticBatch


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: Optional[torch.optim.Optimizer] = None,
        loss_fn: Callable = np_loss_fn,
        checkpointer: Optional[ModelCheckpointer] = None,
        plot_interval: int = 1,
    ):
        super().__init__()

        self.model = model
        self.optimiser = (
            optimiser if optimiser is not None else torch.optim.Adam(model.parameters())
        )
        self.loss_fn = loss_fn
        self.checkpointer = checkpointer
        self.plot_interval = plot_interval
        self.val_outputs: List[Any] = []
        self.test_outputs: List[Any] = []
        self.train_losses: List[Any] = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        _ = batch_idx
        loss = self.loss_fn(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.detach().cpu())
        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": batch}
        if isinstance(batch, ICBatch):
            pred_dist = self.model(
                xc=batch.xc,
                yc=batch.yc,
                xic=batch.xic,
                yic=batch.yic,
                xt=batch.xt,
            )
        elif isinstance(batch, GriddedImageBatch):
            pred_dist = self.model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
        else:
            pred_dist = self.model(xc=batch.xc, yc=batch.yc, xt=batch.xt)

        loglik = pred_dist.log_prob(batch.yt).mean()
        result["loglik"] = loglik.cpu()

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.mean() / batch.yt.shape[1]
            result["gt_loglik"] = gt_loglik.cpu()

        self.val_outputs.append(result)

    def test_step(  # pylint: disable=arguments-differ
        self, batch: Batch, batch_idx: int
    ) -> None:
        _ = batch_idx
        result = {"batch": _batch_to_cpu(batch)}
        if isinstance(batch, ICBatch):
            pred_dist = self.model(
                xc=batch.xc,
                yc=batch.yc,
                xic=batch.xic,
                yic=batch.yic,
                xt=batch.xt,
            )
        elif isinstance(batch, GriddedImageBatch):
            pred_dist = self.model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
        else:
            pred_dist = self.model(xc=batch.xc, yc=batch.yc, xt=batch.xt)

        loglik = pred_dist.log_prob(batch.yt).mean()
        result["loglik"] = loglik.cpu()

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.mean() / batch.yt.shape[1]
            result["gt_loglik"] = gt_loglik.cpu()

        self.test_outputs.append(result)

    def on_train_epoch_end(self) -> None:
        train_losses = torch.stack(self.train_losses)
        self.train_losses = []

        if self.checkpointer is not None:
            # For checkpointing.
            train_result = {
                "mean_loss": train_losses.mean(),
                "std_loss": train_losses.std() / (len(train_losses) ** 0.5),
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model,
                val_result=train_result,
                prefix="train_",
                update_last=False,
            )

    def on_validation_epoch_end(self) -> None:
        results = {
            k: [result[k] for result in self.val_outputs]
            for k in self.val_outputs[0].keys()
        }
        self.val_outputs = []

        loglik = torch.stack(results["loglik"])
        mean_loglik = loglik.mean()
        std_loglik = loglik.std() / (len(loglik) ** 0.5)
        self.log("val/loglik", mean_loglik)
        self.log("val/std_loglik", std_loglik)

        if self.checkpointer is not None:
            # For checkpointing.
            val_result = {
                "mean_loss": -mean_loglik,
                "std_loss": std_loglik,
            }
            self.checkpointer.update_best_and_last_checkpoint(
                model=self.model, val_result=val_result, prefix="val_"
            )

        if "gt_loglik" in results:
            gt_loglik = torch.stack(results["gt_loglik"])
            mean_gt_loglik = gt_loglik.mean()
            std_gt_loglik = gt_loglik.std() / (len(gt_loglik) ** 0.5)
            self.log("val/gt_loglik", mean_gt_loglik)
            self.log("val/std_gt_loglik", std_gt_loglik)

        if self.current_epoch % self.plot_interval == 0:
            if isinstance(self.trainer.val_dataloaders, ImageGenerator):
                plot_image(
                    model=self.model,
                    batches=results["batch"],
                    num_fig=min(5, len(results["batch"])),
                    name=f"epoch-{self.current_epoch:04d}",
                )
            elif isinstance(self.trainer.val_dataloaders, CRUDataGenerator):
                plot_cru(
                    model=self.model,
                    batches=results["batch"],
                    x_mean=self.trainer.val_dataloaders.x_mean,
                    x_std=self.trainer.val_dataloaders.x_std,
                    y_mean=self.trainer.val_dataloaders.y_mean,
                    y_std=self.trainer.val_dataloaders.y_std,
                    num_fig=min(5, len(results["batch"])),
                    figsize=(24.0, 5.0),
                    lat_range=self.trainer.val_dataloaders.lat_range,
                    lon_range=self.trainer.val_dataloaders.lon_range,
                    time_idx=[0, -1],
                    name=f"epoch-{self.current_epoch:04d}",
                )
            elif isinstance(self.trainer.val_dataloaders, KolmogorovGenerator):
                plot_kolmogorov(
                    model=self.model,
                    batches=results["batch"],
                    num_fig=min(5, len(results["batch"])),
                    figsize=(18.0, 5.0),
                    subplots=True,
                    name=f"epoch-{self.current_epoch:04d}",
                )
            else:
                plot(
                    model=self.model,
                    batches=results["batch"],
                    num_fig=min(5, len(results["batch"])),
                    name=f"epoch-{self.current_epoch:04d}",
                )

    def configure_optimizers(self):
        return self.optimiser


def _batch_to_cpu(batch: Batch):
    batch_kwargs = {
        field.name: getattr(batch, field.name).cpu()
        if isinstance(getattr(batch, field.name), torch.Tensor)
        else getattr(batch, field.name)
        for field in dataclasses.fields(batch)
    }
    return type(batch)(**batch_kwargs)
