from collections import defaultdict
from functools import partial
from typing import Tuple

import einops
import gpytorch
import torch
from gp_utils import train_gp
from plot import plot
from plot_cru import plot_cru
from plot_kolmogorov import plot_kolmogorov
from tqdm.auto import tqdm
from utils import initialize_experiment

import wandb
from icicl.data.cru import CRUDataGenerator
from icicl.data.kolmogorov import KolmogorovGenerator


def optimise_gp(
    init_model: partial[gpytorch.Module],
    init_likelihood: partial[gpytorch.likelihoods.Likelihood],
    init_optimiser: partial[torch.optim.Optimizer],
    xc: torch.Tensor,
    yc: torch.Tensor,
    **kwargs,
) -> Tuple[gpytorch.Module, gpytorch.likelihoods.Likelihood, dict]:
    # GPytorch requires these to have no output dimension.
    yc = einops.rearrange(yc, "n 1 -> n")

    # Initialise and train the model.
    likelihood = init_likelihood()
    model = init_model(train_x=xc, train_y=yc, likelihood=likelihood)
    model.train()
    likelihood.train()

    optimiser = init_optimiser(model.parameters())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll.train()
    train_result = train_gp(
        model=model,
        mll=mll,
        optimiser=optimiser,
        xc=xc,
        yc=yc,
        **kwargs,
    )
    return model, likelihood, train_result


def optimise_gp_and_return_pred_dist(
    init_model: partial[gpytorch.Module],
    init_likelihood: partial[gpytorch.likelihoods.Likelihood],
    init_optimiser: partial[torch.optim.Optimizer],
    iters: int,
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    **kwargs,
) -> torch.distributions.Distribution:
    if len(xc.shape) == 3:
        assert (
            xc.shape[0] == 1 and yc.shape[0] == 1 and xt.shape[0] == 1
        ), "Can only accept a batch size of 1."
        xc = einops.rearrange(xc, "1 n d -> n d")
        yc = einops.rearrange(yc, "1 n 1 -> n 1")
        xt = einops.rearrange(xt, "1 n d -> n d")

    model, likelihood, _ = optimise_gp(
        init_model=init_model,
        init_likelihood=init_likelihood,
        init_optimiser=init_optimiser,
        xc=xc,
        yc=yc,
        iters=iters,
        **kwargs,
    )

    # Now get predictive log-likelihood.
    model.eval()
    likelihood.eval()
    output = likelihood(model(xt))
    pred_mean = einops.rearrange(output.mean, "n -> n 1")
    pred_var = einops.rearrange(output.variance, "n -> n 1")
    pred_dist = torch.distributions.Normal(pred_mean, pred_var**0.5)
    return pred_dist


def main():
    experiment, _ = initialize_experiment()

    init_likelihood = experiment.likelihood
    init_model = experiment.model
    init_optimiser = experiment.optimiser
    gen_val = experiment.generators.val
    train_iters = experiment.params.train_iters
    eval_name = experiment.misc.eval_name

    if experiment.misc.only_plots:
        eval_name = experiment.misc.name + "/" + eval_name

        gen_val.batch_size = 1
        gen_val.num_batches = experiment.misc.num_plots
        batches = list(gen_val)

        if isinstance(gen_val, CRUDataGenerator):
            plot_cru(
                model=partial(
                    optimise_gp_and_return_pred_dist,
                    init_model,
                    init_likelihood,
                    init_optimiser,
                    train_iters,
                ),
                batches=batches,
                x_mean=gen_val.x_mean,
                x_std=gen_val.x_std,
                y_mean=gen_val.y_mean,
                y_std=gen_val.y_std,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(6, 6),
                # lat_range=gen_val.lat_range,
                # lon_range=gen_val.lon_range,
                time_idx=[0],
                name=eval_name,
                subplots=experiment.misc.subplots,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
            )
        elif isinstance(gen_val, KolmogorovGenerator):
            plot_kolmogorov(
                model=partial(
                    optimise_gp_and_return_pred_dist,
                    init_model,
                    init_likelihood,
                    init_optimiser,
                    train_iters,
                ),
                batches=batches,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(18.0, 5.0),
                plot_dims=(0, 1),
                other_dim_slice=0,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
                subplots=experiment.misc.subplots,
            )
        else:
            plot(
                model=partial(
                    optimise_gp_and_return_pred_dist,
                    init_model,
                    init_likelihood,
                    init_optimiser,
                    train_iters,
                ),
                batches=batches,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(6, 5),
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
            )
        return

    val_tqdm = tqdm(gen_val, total=len(gen_val), desc="Outer loop")
    dataset_idx = 0
    results = defaultdict(list)
    for batch in val_tqdm:
        for xc, yc, xt, yt in zip(batch.xc, batch.yc, batch.xt, batch.yt):
            pred_dist = optimise_gp_and_return_pred_dist(
                init_model=init_model,
                init_likelihood=init_likelihood,
                init_optimiser=init_optimiser,
                iters=train_iters,
                xc=xc,
                yc=yc,
                xt=xt,
                dataset_idx=dataset_idx,
                log=(
                    (dataset_idx % experiment.misc.log_interval == 0)
                    and experiment.misc.logging
                ),
            )

            loglik = pred_dist.log_prob(yt)

            results["loglik"].append(loglik.mean())

            # Log moving average loglik.
            mean_loglik = torch.stack(results["loglik"]).mean()
            wandb.log({f"{eval_name}/loglik_ma": mean_loglik})

            dataset_idx += 1

    loglik = torch.stack(results["loglik"])
    mean_loglik = loglik.mean()
    wandb.run.summary[f"eval/{eval_name}/mean_loglik"] = loglik.mean()
    wandb.run.summary[f"eval/{eval_name}/std_loglik"] = loglik.std() / (
        len(loglik) ** 0.5
    )


if __name__ == "__main__":
    main()
