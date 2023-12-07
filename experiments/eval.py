import lightning.pytorch as pl
from plot import plot
from plot_image import plot_image
from utils import initialize_evaluation, val_epoch

import wandb
from icicl.data.image import ImageGenerator


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    gen_val = experiment.generators.val

    # Store number of parameters.
    num_params = sum(p.numel() for p in model.parameters())
    wandb.run.summary["num_params"] = num_params

    # Store test set performance.
    pl.seed_everything(0)
    test_result, batches = val_epoch(model=model, generator=gen_val)
    wandb.run.summary["test/loglik"] = test_result["mean_loglik"]
    wandb.run.summary["test/std_loglik"] = test_result["std_loglik"]
    if "mean_gt_loglik" in test_result:
        wandb.run.summary["test/gt_loglik"] = test_result["mean_gt_loglik"]
        wandb.run.summary["test/std_gt_loglik"] = test_result["std_gt_loglik"]

    if isinstance(gen_val, ImageGenerator):
        plot_image(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name="test",
        )
    else:
        plot(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name="test",
        )


if __name__ == "__main__":
    main()