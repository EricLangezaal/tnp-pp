"""https://github.com/cambridge-mlg/dpconvcnp/blob/main/experiments/utils.py"""

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from check_shapes import check_shapes
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import wandb
from icicl.data.data import Batch, DataGenerator


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

        loss = loss_fn(
            model=model,
            xc=batch.xc,
            yc=batch.yc,
            xt=batch.xt,
            yt=batch.yt,
            xic=batch.xic,
            yic=batch.yic,
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
            if batch.xic is not None and batch.yic is not None:
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
        gt_loglik = gt_loglik.mean()

        result["loglik"].append(loglik)
        result["pred_mean"].append(pred_dist.loc)
        result["pred_std"].append(pred_dist.scale)
        result["gt_mean"].append(gt_mean)
        result["gt_std"].append(gt_std)
        result["gt_loglik"].append(gt_loglik)

    loss = -torch.stack(result["loglik"]).mean()
    wandb.log({"val/loss": loss, "epoch": epoch})

    return result, batches


def initialize_experiment() -> Tuple[Any, DictConfig]:
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
        project=experiment.misc.project, name=experiment.misc.name, config=config_dict
    )

    return experiment


def evaluation_summary(name: str, result: Dict[str, Any]) -> None:
    wandb.log(
        {
            f"{name}/loglik": torch.stack(result["loglik"]).mean(),
            f"{name}/gt_loglik": torch.stack(result["gt_loglik"]).mean(),
        }
    )
