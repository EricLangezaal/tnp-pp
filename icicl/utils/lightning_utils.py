import dataclasses
from typing import Any, Callable, List, Optional

import lightning.pytorch as pl
import torch
from torch import nn

from icicl.data.base import Batch
from icicl.utils.experiment_utils import ModelCheckpointer, np_loss_fn, np_pred_fn


class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimiser: Optional[torch.optim.Optimizer] = None,
        loss_fn: Callable = np_loss_fn,
        pred_fn: Callable = np_pred_fn,
        plot_fn: Optional[Callable] = None,
        checkpointer: Optional[ModelCheckpointer] = None,
        plot_interval: int = 1,
    ):
        super().__init__()

        self.model = model
        self.optimiser = (
            optimiser if optimiser is not None else torch.optim.Adam(model.parameters())
        )
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn
        self.plot_fn = plot_fn
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
        pred_dist = self.pred_fn(self.model, batch)
        loglik = pred_dist.log_prob(batch.yt).mean()
        result["loglik"] = loglik.cpu()

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
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
        pred_dist = self.pred_fn(self.model, batch)
        loglik = pred_dist.log_prob(batch.yt).mean()
        result["loglik"] = loglik.cpu()

        if hasattr(batch, "gt_pred") and batch.gt_pred is not None:
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

        if self.plot_fn is not None and self.current_epoch % self.plot_interval == 0:
            self.plot_fn(
                self.model, results["batch"], f"epoch-{self.current_epoch:04d}"
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
