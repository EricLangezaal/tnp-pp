"""https://github.com/cambridge-mlg/dpconvcnp/blob/main/experiments/utils.py"""

import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from check_shapes import check_shapes
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import wandb
from icicl.data.data import Batch, DataGenerator


class ModelCheckpointer:
    def __init__(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir is None:
            checkpoint_dir = f"{wandb.run.dir}/checkpoints"

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.best_validation_loss = float("inf")

    def update_best_and_last_checkpoint(
        self,
        model: nn.Module,
        val_result: Dict[str, torch.Tensor],
    ) -> None:
        """Update the best and last checkpoints of the model.

        Arguments:
            model: model to save.
            val_result: validation result dictionary.
        """

        loss_ci = val_result["mean_loss"] + 1.96 * val_result["std_loss"]

        if loss_ci < self.best_validation_loss:
            self.best_validation_loss = loss_ci
            torch.save(
                model.state_dict(), os.path.join(self.checkpoint_dir, "best.ckpt")
            )

        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, "last.ckpt"))

    def load_best_checkpoint(
        self, model: nn.Module, path: Optional[str] = None
    ) -> None:
        if path is None:
            path = torch.load(os.path.join(self.checkpoint_dir, "best.ckpt"))

        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"Checkpoint file {path} not found.")

    def load_last_checkpoint(
        self, model: nn.Module, path: Optional[str] = None
    ) -> None:
        if path is None:
            path = torch.load(os.path.join(self.checkpoint_dir, "last.ckpt"))

        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        else:
            raise FileNotFoundError(f"Checkpoint file {path} not found.")


@check_shapes(
    "xc: [m, nc, dx]",
    "yc: [m, nc, dy]",
    "xt: [m, nt, dx]",
    "yt: [m, nt, dy]",
    "xic: [m, nic, ncic, dx]",
    "yic: [m, nic, ncic, dy]",
    "return: []",
)
def np_loss_fn(
    model: nn.Module,
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    yt: torch.Tensor,
    xic: Optional[torch.Tensor] = None,
    yic: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        xc: Tensor representing context inputs.
        yc: Tensor representing context outputs.
        xt: Tensor representing target inputs.
        yt: Tensor representing target outputs.
        xic: Optional[Tensor] representing in-context inputs.
        yic: Optional[Tensor] representing in-context outputs.
        optimizer: optimizer to use in the training step.

    Returns:
        loss: average negative log likelihood.
    """
    if xic is not None and yic is not None:
        pred_dist = model(xc, yc, xic, yic, xt)
    else:
        pred_dist = model(xc, yc, xt)

    loglik = pred_dist.log_prob(yt)
    return -loglik.mean()


def train_epoch(
    model: nn.Module,
    generator: DataGenerator,
    optimiser: torch.optim.Optimizer,
    step: int,
    loss_fn: Callable = np_loss_fn,
) -> int:
    epoch = tqdm(generator, total=generator.num_batches, desc="Training")

    for batch in epoch:
        optimiser.zero_grad()

        if not hasattr(batch, "xic") or not hasattr(batch, "yic"):
            xic = None
            yic = None
        else:
            xic = batch.xic
            yic = batch.yic

        loss = loss_fn(
            model=model,
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            yt=batch.yt,
            xic=xic,
            yic=yic,
        )
        loss.backward()
        optimiser.step()

        wandb.log({"train/loss": loss, "step": step})
        epoch.set_postfix({"train/loss": loss.item()})

        step += 1

    return step


def val_epoch(
    model: nn.Module,
    generator: DataGenerator,
    epoch: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Batch]]:
    result: Dict = {
        "loglik": [],
        "pred_mean": [],
        "pred_std": [],
        "gt_mean": [],
        "gt_std": [],
        "gt_loglik": [],
    }
    batches = []

    for batch in tqdm(generator, total=generator.num_batches, desc="Validation"):
        batches.append(batch)
        with torch.no_grad():
            if hasattr(batch, "xic") and hasattr(batch, "yic"):
                pred_dist = model(
                    xc=batch.xc,
                    yc=batch.yc,
                    xic=batch.xic,
                    yic=batch.yic,
                    xt=batch.xt,
                )
            else:
                pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)

        loglik = pred_dist.log_prob(batch.yt).mean()

        gt_mean, gt_std, gt_loglik = batch.gt_pred(
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            yt=batch.yt,
        )
        gt_loglik = gt_loglik.mean() / batch.yt.shape[1]

        result["loglik"].append(loglik)
        result["pred_mean"].append(pred_dist.loc)
        result["pred_std"].append(pred_dist.scale)
        result["gt_mean"].append(gt_mean)
        result["gt_std"].append(gt_std)
        result["gt_loglik"].append(gt_loglik)

    loglik = torch.stack(result["loglik"])
    loss = -loglik.mean()
    wandb.log({"val/loss": loss, "epoch": epoch})

    result["mean_loss"] = -loglik.mean()
    result["std_loss"] = -loglik.std()

    return result, batches


def initialize_experiment() -> Tuple[DictConfig, ModelCheckpointer]:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, config_changes = parser.parse_known_args()

    # Initialise experiment, make path.
    OmegaConf.register_new_resolver("eval", eval)
    config = OmegaConf.load(args.config)
    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    experiment = instantiate(config)

    # Initialise wandb.
    wandb.init(
        project=experiment.misc.project,
        name=experiment.misc.name,
        config=config_dict,
    )

    checkpointer = ModelCheckpointer()
    if experiment.misc.resume_from_checkpoint is not None:
        checkpointer.load_best_checkpoint(
            experiment.model, experiment.misc.resume_from_checkpoint
        )

    return experiment, checkpointer


def evaluation_summary(name: str, result: Dict[str, Any]) -> None:
    wandb.log(
        {
            f"{name}/loglik": torch.stack(result["loglik"]).mean(),
            f"{name}/gt_loglik": torch.stack(result["gt_loglik"]).mean(),
        }
    )
