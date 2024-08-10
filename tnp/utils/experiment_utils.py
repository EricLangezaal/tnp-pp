import argparse
import os
import resource
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partialmethod

import einops
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

import wandb
from tnp.data.on_off_grid import DataModality
from tnp.data.era5 import ERA5DataGenerator
from tnp.data.base import Batch, DataGenerator
from tnp.data.synthetic import SyntheticBatch
from tnp.data.on_off_grid import OOTGBatch
from tnp.models.base import ConditionalNeuralProcess, NeuralProcess, OOTGConditionalNeuralProcess
from tnp.models.convcnp import GriddedConvCNP
from tnp.utils.batch import compress_batch_dimensions


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
                if prefix is not None:
                    name = f"{prefix}best.ckpt"
                else:
                    name = "best.ckpt"

                torch.save(
                    model.state_dict(),
                    os.path.join(self.checkpoint_dir, name),
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


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    #print(model)
    if isinstance(model, OOTGConditionalNeuralProcess):
        assert isinstance(batch, OOTGBatch)
        pred_dist = model(
            xc_off_grid = batch.xc_off_grid, 
            yc_off_grid = batch.yc_off_grid, 
            xc_on_grid = batch.xc_on_grid,
            yc_on_grid = batch.yc_on_grid,
            xt = batch.xt,
            used_modality = batch.used_modality,
        )
    elif isinstance(model, GriddedConvCNP):
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, NeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    else:
        raise ValueError

    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.distributions.Distribution:
    if isinstance(model, OOTGConditionalNeuralProcess):
        assert isinstance(batch, OOTGBatch)
        pred_dist = model(
            xc_off_grid = batch.xc_off_grid, 
            yc_off_grid = batch.yc_off_grid, 
            xc_on_grid = batch.xc_on_grid,
            yc_on_grid = batch.yc_on_grid,
            xt = batch.xt,
            used_modality = batch.used_modality,
        )
    elif isinstance(model, GriddedConvCNP):
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, NeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    else:
        raise ValueError

    return pred_dist


def train_epoch(
    model: nn.Module,
    generator: DataGenerator,
    optimiser: torch.optim.Optimizer,
    step: int,
    loss_fn: Callable = np_loss_fn,
    gradient_clip_val: Optional[float] = None,
) -> Tuple[int, Dict[str, Any]]:
    epoch = tqdm(generator, total=len(generator), desc="Training")
    losses = []
    for batch in epoch:
        optimiser.zero_grad()
        loss = loss_fn(model=model, batch=batch)
        loss.backward()

        if gradient_clip_val is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

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
        "mean_loss": -loglik.mean(),
        "std_loss": loglik.std() / (len(losses) ** 0.5),
    }

    return step, train_result


def val_epoch(
    model: nn.Module,
    generator: DataGenerator,
    num_samples: int = 1,
) -> Tuple[Dict[str, Any], List[Batch]]:
    result = defaultdict(list)
    batches = []

    for batch in tqdm(generator, total=len(generator), desc="Validation"):
        batches.append(batch)
        with torch.no_grad():
            if isinstance(model, OOTGConditionalNeuralProcess):
                assert isinstance(batch, OOTGBatch)
                pred_dist = model(
                    xc_off_grid = batch.xc_off_grid, 
                    yc_off_grid = batch.yc_off_grid, 
                    xc_on_grid = batch.xc_on_grid,
                    yc_on_grid = batch.yc_on_grid,
                    xt = batch.xt,
                    used_modality = batch.used_modality,
                )
            elif isinstance(model, GriddedConvCNP):
                pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
            elif isinstance(model, ConditionalNeuralProcess):
                pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
            elif isinstance(model, NeuralProcess):
                pred_dist = model(
                    xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
                )
            else:
                raise ValueError

        loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

        if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
            gt_mean, gt_std, gt_loglik = batch.gt_pred(
                xc=batch.xc,
                yc=batch.yc,
                xt=batch.xt,
                yt=batch.yt,
                batch=batch,
            )
            gt_loglik = gt_loglik.sum() / batch.yt[..., 0].numel()

            result["gt_mean"].append(gt_mean)
            result["gt_std"].append(gt_std)
            result["gt_loglik"].append(gt_loglik)

        result["loglik"].append(loglik)
        if isinstance(generator, ERA5DataGenerator):
            rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()
            result["rmse"].append(generator.y_std[0] * rmse)


    loglik = torch.stack(result["loglik"])
    result["mean_loglik"] = loglik.mean()
    result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)
    result["mean_loss"] = -loglik.mean()
    result["std_loss"] = loglik.std() / (len(loglik) ** 0.5)

    if "rmse" in result:
        result["rmse"] = torch.stack(result["rmse"]).mean()

    if "gt_loglik" in result:
        gt_loglik = torch.stack(result["gt_loglik"])
        result["mean_gt_loglik"] = gt_loglik.mean()
        result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

    return result, batches


def create_default_config() -> DictConfig:
    default_config = {
        "misc": {
            "resume_from_checkpoint": None,
            "resume_from_path": None,
            "override_config": False,
            "plot_ar_mode": False,
            "logging": True,
            "progress_bars": True,
            "seed": 0,
            "plot_interval": 1,
            "lightning_eval": True,
            "num_plots": 5,
            "gradient_clip_val": None,
            "only_plots": False,
            "fake_train_steps": None,
            "savefig": False,
            "subplots": False,
            "loss_fn": {
                "_target_": "tnp.utils.experiment_utils.np_loss_fn",
                "_partial_": True,
            },
            "pred_fn": {
                "_target_": "tnp.utils.experiment_utils.np_pred_fn",
                "_partial_": True,
            },
        }
    }
    return OmegaConf.create(default_config)


def extract_config(
    raw_config: Union[str, Dict],
    config_changes: Optional[List[str]] = None,
    combine_default: bool = True,
) -> Tuple[DictConfig, Dict]:
    """Extract the config from the config file and the config changes.

    Arguments:
        config_file: path to the config file.
        config_changes: list of config changes.

    Returns:
        config: config object.
        config_dict: config dictionary.
    """
    # Register eval.
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    if isinstance(raw_config, str):
        config = OmegaConf.load(raw_config)
    else:
        config = OmegaConf.create(raw_config)

    if combine_default:
        default_config = create_default_config()
        config = OmegaConf.merge(default_config, config)

    config_changes = OmegaConf.from_cli(config_changes)
    config = OmegaConf.merge(config, config_changes)
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config, config_dict


def deep_convert_dict(layer: Any):
    to_ret = layer
    if isinstance(layer, OrderedDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret


def initialize_experiment() -> Tuple[DictConfig, ModelCheckpointer]:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--generator_config", type=str)
    args, config_changes = parser.parse_known_args()

    if args.generator_config is not None:
        # Merge generator config with config.
        raw_config = deep_convert_dict(
            hiyapyco.load(
                (args.config, args.generator_config),
                method=hiyapyco.METHOD_MERGE,
                usedefaultyamlloader=True,
            )
        )
    else:
        raw_config = args.config

    # Initialise experiment, make path.
    config, config_dict = extract_config(raw_config, config_changes)

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
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    pl.seed_everything(experiment.misc.seed)

    # use tensor cores effectively
    torch.set_float32_matmul_precision('high')

    # allow opening crazy number of files
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(hard_lim, hard_lim))

    if isinstance(experiment.model, nn.Module):
        if experiment.misc.resume_from_checkpoint:
            # Downloads to "./checkpoints/last.ckpt".
            ckpt_file = run.files("checkpoints/last.ckpt")[0]
            ckpt_file.download(replace=True)
            experiment.model.load_state_dict(
                torch.load("checkpoints/last.ckpt", map_location="cpu"), strict=True
            )

        elif experiment.misc.resume_from_path is not None:
            experiment.model.load_state_dict(
                torch.load(experiment.misc.resume_from_path, map_location="cpu"),
                strict=True,
            )
    else:
        print("Did not initialise as nn.Module.")

    # Initialise wandb. Set logging: True if wandb logging needed.
    if experiment.misc.logging:
        wandb.init(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=config_dict,
        )

    if not experiment.misc.progress_bars:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    checkpointer = ModelCheckpointer(logging=experiment.misc.logging)

    return experiment, checkpointer


def initialize_evaluation() -> DictConfig:
    # Make argument parser with config argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--ckpt", type=str, choices=["val_best", "train_best", "last"], default="last"
    )
    args, config_changes = parser.parse_known_args()

    # only do this if previous run specified
    if args.run_path is not None:
        api = wandb.Api()
        run = api.run(args.run_path)

    # Initialise evaluation, make path.
    config, config_dict = extract_config(args.config, config_changes)

    # Set model to run.config.model.
    if args.run_path is not None:
        #config.model = run.config["model"]
        pass

    # Set random seed.
    pl.seed_everything(config.misc.seed)

    # Instantiate.
    experiment = instantiate(config)

    # Set random seed.
    pl.seed_everything(config.misc.seed)

    if args.run_path is not None:
        # Downloads to "./checkpoints/last.ckpt"
        ckpt_file = run.files(f"checkpoints/{args.ckpt}.ckpt")[0]
        ckpt_file.download(replace=True)

        experiment.model.load_state_dict(
            torch.load(f"checkpoints/{args.ckpt}.ckpt", map_location="cpu"), strict=True
        )
        wandb.init(
            resume="must",
            project=run.project,
            name=run.name,
            id=run.id,
        )
    else:
        wandb.init(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=config_dict
        )

    return experiment


def evaluation_summary(name: str, result: Dict[str, Any]) -> None:
    if wandb.run is None:
        return

    if "mean_loglik" in result:
        wandb.log({f"{name}/loglik": result["mean_loglik"]})

    if "rmse" in result:
        wandb.log({f"{name}/rmse": result["rmse"]})

    if "mean_gt_loglik" in result:
        wandb.log(
            {
                f"{name}/gt_loglik": result["mean_gt_loglik"],
            }
        )
