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
    eps = experiment.misc.eps

    # Update context and target range with eps.
    gen_val.context_range = gen_val.context_range + eps
    gen_val.target_range = gen_val.target_range + eps

    # Store test set performance.
    pl.seed_everything(0)
    test_result, batches = val_epoch(model=model, generator=gen_val)
    wandb.run.summary[f"test/te/{eps}/loglik"] = test_result["mean_loglik"]
    wandb.run.summary[f"test/te/{eps}/std_loglik"] = test_result["std_loglik"]

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
            name=f"test/te/{eps}",
        )


if __name__ == "__main__":
    main()
