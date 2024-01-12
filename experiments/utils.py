"""https://github.com/cambridge-mlg/dpconvcnp/blob/main/experiments/utils.py"""

import argparse
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import einops
import lightning.pytorch as pl
import torch
from check_shapes import check_shapes
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import wandb
from icicl.data.base import Batch, DataGenerator, ICBatch
from icicl.data.synthetic import SyntheticBatch
from icicl.utils.batch import compress_batch_dimensions
from icicl.utils.initialisation import weights_init


class ModelCheckpointer:
    def __init__(self, checkpoint_dir: Optional[str] = None, logging: bool = True):
        self.logging = logging

        self.checkpoint_dir: Optional[str] = None

        if checkpoint_dir is None and logging:
            checkpoint_dir = f"{wandb.run.dir}/checkpoints"

            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            self.checkpoint_dir = checkpoint_dir

        self.best_validation_loss = float("inf")

    def update_best_and_last_checkpoint(
        self,
        model: nn.Module,
        val_result: Dict[str, torch.Tensor],
        prefix: Optional[str] = None,
        update_last: bool = True,
    ) -> None:
        """Update the best and last checkpoints of the model.

        Arguments:
            model: model to save.
            val_result: validation result dictionary.
        """

        loss_ci = val_result["mean_loss"] + 1.96 * val_result["std_loss"]

        if loss_ci < self.best_validation_loss:
            self.best_validation_loss = loss_ci
            if self.logging:
                assert self.checkpoint_dir is not None
                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, f"{prefix}best.ckpt"),
                )

        if update_last and self.logging:
            assert self.checkpoint_dir is not None
            torch.save(
                model.state_dict(), os.path.join(self.checkpoint_dir, "last.ckpt")
            )

    def load_best_checkpoint(
        self, model: nn.Module, path: Optional[str] = None
    ) -> None:
        if path is None and self.logging:
            assert self.checkpoint_dir is not None
            path = os.path.join(self.checkpoint_dir, "best.ckpt")
        elif path is None:
            raise ValueError("Not logging to any checkpoints.")

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"Checkpoint file {path} not found.")

    def load_last_checkpoint(
        self, model: nn.Module, path: Optional[str] = None
    ) -> None:
        if path is None and self.logging:
            assert self.checkpoint_dir is not None
            path = os.path.join(self.checkpoint_dir, "last.ckpt")
        elif path is None:
            raise ValueError("Not logging to any checkpoints.")

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
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
) -> Tuple[int, Dict[str, Any]]:
    epoch = tqdm(generator, total=len(generator), desc="Training")
    losses = []
    for batch in epoch:
        optimiser.zero_grad()

        if isinstance(batch, ICBatch):
            loss = loss_fn(
                model=model,
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                yt=batch.yt,
                xic=batch.xic,
                yic=batch.yic,
            )
        else:
            assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
            loss = loss_fn(
                model=model, xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
            )

        loss.backward()
        optimiser.step()

        losses.append(loss.detach())
        epoch.set_postfix({"train/loss": loss.item()})

        if wandb.run is not None:
            wandb.log({"train/loss": loss, "step": step})

        step += 1

    loglik = -torch.stack(losses)
    train_result = {
        "loglik": loglik,
        "mean_loglik": loglik.mean(),
        "std_loglik": loglik.std() / (len(losses) ** 0.5),
    }

    return step, train_result


def val_epoch(
    model: nn.Module,
    generator: DataGenerator,
) -> Tuple[Dict[str, Any], List[Batch]]:
    result = defaultdict(list)
    batches = []

    for batch in tqdm(generator, total=len(generator), desc="Validation"):
        batches.append(batch)
        with torch.no_grad():
            if isinstance(batch, ICBatch):
                pred_dist = model(
                    xc=batch.xc,
                    yc=batch.yc,
                    xic=batch.xic,
                    yic=batch.yic,
                    xt=batch.xt,
                )
            else:
                assert not hasattr(batch, "xic") and not hasattr(batch, "yic")
                pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)

        loglik = pred_dist.log_prob(batch.yt).mean()

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            gt_mean, gt_std, gt_loglik = batch.gt_pred(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                yt=batch.yt,
            )
            gt_loglik = gt_loglik.mean() / batch.yt.shape[1]

            result["gt_mean"].append(gt_mean)
            result["gt_std"].append(gt_std)
            result["gt_loglik"].append(gt_loglik)

        result["loglik"].append(loglik)
        result["pred_mean"].append(pred_dist.loc)
        result["pred_std"].append(pred_dist.scale)

    loglik = torch.stack(result["loglik"])
    result["mean_loglik"] = loglik.mean()
    result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)
    result["mean_loss"] = -loglik.mean()
    result["std_loss"] = loglik.std() / (len(loglik) ** 0.5)

    if "gt_loglik" in result:
        gt_loglik = torch.stack(result["gt_loglik"])
        result["mean_gt_loglik"] = gt_loglik.mean()
        result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

    return result, batches


def create_default_config() -> DictConfig:
    default_config = {
        "misc": {
            "resume_from_checkpoint": None,
            "override_config": False,
            "plot_ar_mode": False,
            "logging": True,
            "seed": 0,
        }
    }
    return OmegaConf.create(default_config)


def extract_config(
    config_file: str, config_changes: List[str]
) -> Tuple[DictConfig, Dict]:
    """Extract the config from the config file and the config changes.

    Arguments:
        config_file: path to the config file.
        config_changes: list of config changes.

    Returns:
        config: config object.
        config_dict: config dictionary.
    """
    default_config = create_default_config()
    OmegaConf.register_new_resolver("eval", eval)
    config = OmegaConf.load(config_file)
    config = OmegaConf.merge(default_config, config)
    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config, config_dict


def initialize_experiment() -> Tuple[DictConfig, ModelCheckpointer]:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, config_changes = parser.parse_known_args()

    # Initialise experiment, make path.
    config, config_dict = extract_config(args.config, config_changes)

    # Get run and potentially override config before instantiation.
    if config.misc.resume_from_checkpoint is not None:
        # Downloads to "./checkpoints/last.ckpt".
        api = wandb.Api()
        run = api.run(config.misc.resume_from_checkpoint)

        # Overide config if specified.
        if config.misc.override_config:
            config = OmegaConf.create(run.config)
            config_dict = run.config

    # Instantiate experiment and load checkpoint.
    experiment = instantiate(config)
    pl.seed_everything(experiment.misc.seed)

    if experiment.misc.resume_from_checkpoint:
        # Downloads to "./checkpoints/last.ckpt".
        ckpt_file = run.files("checkpoints/last.ckpt")[0]
        ckpt_file.download(replace=True)
        experiment.model.load_state_dict(
            torch.load("checkpoints/last.ckpt", map_location="cpu")
        )

    else:
        # Initialise model weights.
        weights_init(experiment.model)

    # Initialise wandb. Set logging: True if wandb logging needed.
    if experiment.misc.logging:
        wandb.init(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=config_dict,
        )

    checkpointer = ModelCheckpointer(logging=experiment.misc.logging)

    return experiment, checkpointer


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--ckpt", type=str, choices=["best", "last"], default="last")
    args, config_changes = parser.parse_known_args()

    api = wandb.Api()
    run = api.run(args.run_path)
    config = run.config

    # Initialise experiment, make path.
    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Instantiate.
    experiment = instantiate(config)

    # Set random seed.
    pl.seed_everything(config.misc.seed)

    # Downloads to "./checkpoints/last.ckpt"
    ckpt_file = run.files(f"checkpoints/{args.ckpt}.ckpt")[0]
    ckpt_file.download(replace=True)

    experiment.model.load_state_dict(
        torch.load(f"checkpoints/{args.ckpt}.ckpt", map_location="cpu")
    )

    # Initialise wandb.
    wandb.init(
        resume="must",
        project=experiment.misc.project,
        name=experiment.misc.name,
        id=run.id,
        config=config_dict,
    )

    return experiment


def evaluation_summary(name: str, result: Dict[str, Any]) -> None:
    if wandb.run is None:
        return

    if "mean_loglik" in result:
        wandb.log({f"{name}/loglik": result["mean_loglik"]})

    if "mean_gt_loglik" in result:
        wandb.log(
            {
                f"{name}/gt_loglik": result["mean_gt_loglik"],
            }
        )


def ar_predict(
    model: nn.Module,
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples from the joint predictive probability distribution. Assumes order of xt is given.

    Args:
        model (nn.Module): NeuralProcess model.
        xc (torch.Tensor): Context inputs.
        yc (torch.Tensor): Context outputs.
        xt (torch.Tensor): Target inputs.
        yt (torch.Tensor): Target outputs.
        num_samples (int): Number of predictive samples to generate and use to estimate likelihood.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Samples and log probabilities drawn from the joint distribution.
    """

    samples_list: List[torch.Tensor] = []
    sample_logprobs_list: List[torch.Tensor] = []

    # Expand tensors for efficient computation.
    xc_ = einops.repeat(xc, "m n d -> s m n d", s=num_samples)
    yc_ = einops.repeat(yc, "m n d -> s m n d", s=num_samples)
    xt_ = einops.repeat(xt, "m n d -> s m n d", s=num_samples)
    xc_, _ = compress_batch_dimensions(xc_, other_dims=2)
    yc_, _ = compress_batch_dimensions(yc_, other_dims=2)
    xt_, _ = compress_batch_dimensions(xt_, other_dims=2)

    # AR mode for loop.
    for i in range(xt_.shape[1]):
        with torch.no_grad():
            # Compute conditional distribution, sample and evaluate log probabilities.
            pred_dist = model(xc=xc_, yc=yc_, xt=xt_[:, i : i + 1])
            pred = pred_dist.rsample()
            pred_logprob = pred_dist.log_prob(pred)

            # Store samples and probabilities.
            pred = pred.detach()
            pred_logprob = pred_logprob.detach()
            samples_list.append(pred)
            sample_logprobs_list.append(pred_logprob)

            # Update context.
            xc_ = torch.cat([xc_, xt_[:, i : i + 1]], dim=1)
            yc_ = torch.cat([yc_, pred], dim=1)

    # Compute log probability of sample.
    samples = torch.cat(samples_list, dim=1)
    sample_logprobs = torch.cat(sample_logprobs_list, dim=1)

    samples = einops.rearrange(samples, "(s m) n d -> s m n d", s=num_samples)
    sample_logprobs = einops.rearrange(
        sample_logprobs, "(s m) n d -> s m n d", s=num_samples
    )
    sample_logprobs = sample_logprobs.mean(0)

    return samples, sample_logprobs
