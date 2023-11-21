from typing import Any, Callable, List

import lightning.pytorch as pl
import torch
from plot import plot
from plot_image import plot_image
from torch import nn
from utils import ModelCheckpointer

from icicl.data.data import Batch, ICBatch, SyntheticBatch
from icicl.data.image import ImageGenerator


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: Callable,
        checkpointer: ModelCheckpointer,
    ):
        super().__init__()

        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.checkpointer = checkpointer
        self.val_outputs: List[Any] = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        _ = batch_idx

        if isinstance(batch, ICBatch):
            loss = self.loss_fn(
                self.model,
                batch.xc,
                batch.yc,
                batch.xt,
                batch.yt,
                batch.xic,
                batch.yic,
            )
        else:
            loss = self.loss_fn(self.model, batch.xc, batch.yc, batch.xt, batch.yt)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
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
        else:
            pred_dist = self.model(xc=batch.xc, yc=batch.yc, xt=batch.xt)

        loglik = pred_dist.log_prob(batch.yt).mean()
        result["loglik"] = loglik

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            _, _, gt_loglik = batch.gt_pred(
                xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )
            gt_loglik = gt_loglik.mean() / batch.yc.shape[1]
            result["gt_loglik"] = gt_loglik

        self.val_outputs.append(result)

    def on_validation_epoch_end(self) -> None:
        results = {
            k: [result[k] for result in self.val_outputs]
            for k in self.val_outputs[0].keys()
        }
        self.val_outputs = []

        loglik = torch.stack(results["loglik"]).mean()
        self.log("val/loss", -loglik)
        self.log("val/loglik", loglik)

        # For checkpointing.
        val_result = {
            "mean_loss": -loglik,
            "std_loss": torch.stack(results["loglik"]).std(),
        }
        self.checkpointer.update_best_and_last_checkpoint(
            model=self.model, val_result=val_result
        )

        if "gt_loglik" in results:
            gt_loglik = torch.stack(results["gt_loglik"]).mean()
            self.log("val/gt_loglik", gt_loglik)

        if isinstance(self.trainer.val_dataloaders, ImageGenerator):
            plot_image(
                model=self.model,
                batches=results["batch"],
                epoch=self.current_epoch,
                num_fig=min(5, len(results["batch"])),
            )
        else:
            plot(
                model=self.model,
                batches=results["batch"],
                epoch=self.current_epoch,
                num_fig=min(5, len(results["batch"])),
            )

    def configure_optimizers(self):
        return self.optimiser
