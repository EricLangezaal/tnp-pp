import gpytorch
import torch
from tqdm.auto import tqdm

import wandb


def train_gp(
    model: gpytorch.Module,
    mll: gpytorch.mlls.MarginalLogLikelihood,
    optimiser: torch.optim.Optimizer,
    xc: torch.Tensor,
    yc: torch.Tensor,
    dataset_idx: int,
    iters: int = 100,
    log: bool = False,
):
    train_iter = tqdm(range(iters), total=iters, desc="Training")
    for _ in train_iter:
        optimiser.zero_grad()
        output = model(xc)
        loss = -mll(output, yc)

        loss.backward()
        optimiser.step()

        train_iter.set_postfix({"train/loss": loss.item()})

        if log:
            wandb.log({f"train/dataset-{dataset_idx}/loss": loss})

    elbo = -loss.detach()
    train_result = {
        "elbo": elbo,
    }

    return train_result
