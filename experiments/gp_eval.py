from collections import defaultdict

import einops
import gpytorch
import lightning.pytorch as pl
import torch
from gp_utils import train_gp
from tqdm.auto import tqdm
from utils import initialize_experiment

import wandb


def main():
    experiment, _ = initialize_experiment()

    init_likelihood = experiment.likelihood
    init_model = experiment.model
    init_optimiser = experiment.optimiser
    gen_val = experiment.generators.val
    train_iters = experiment.params.train_iters
    eval_name = experiment.misc.eval_name

    pl.seed_everything(0)

    val_tqdm = tqdm(gen_val, total=len(gen_val), desc="Outer loop")
    dataset_idx = 0
    results = defaultdict(list)
    for batch in val_tqdm:
        for xc, yc, xt, yt in zip(batch.xc, batch.yc, batch.xt, batch.yt):
            # GPytorch requires these to have no output dimension.
            yc = einops.rearrange(yc, "n 1 -> n")
            yt = einops.rearrange(yt, "n 1 -> n")

            # Initialise and train the model.
            likelihood = init_likelihood()
            model = init_model(train_x=xc, train_y=yc, likelihood=likelihood)
            model.train()
            likelihood.train()

            optimiser = init_optimiser(model.parameters())
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            train_result = train_gp(
                model=model,
                mll=mll,
                optimiser=optimiser,
                xc=xc,
                yc=yc,
                dataset_idx=dataset_idx,
                iters=train_iters,
                log=(dataset_idx % experiment.misc.log_interval == 0),
            )
            results["elbos"].append(train_result["elbo"])

            # Now get predictive log-likelihood.
            model.eval()
            likelihood.eval()
            output = likelihood(model(xt))
            pred_mean = output.mean
            pred_var = output.variance
            pred_dist = torch.distributions.Normal(pred_mean, pred_var**0.5)
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
