import lightning.pytorch as pl
from plot import plot
from plot_image import plot_image
from utils import (
    evaluation_summary,
    initialize_experiment,
    np_loss_fn,
    train_epoch,
    val_epoch,
)

from icicl.data.image import ImageGenerator


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    pl.seed_everything(0)

    step = 0
    for epoch in range(epochs):
        step, train_result = train_epoch(
            model=model,
            generator=gen_train,
            optimiser=optimiser,
            step=step,
            loss_fn=np_loss_fn,
        )
        evaluation_summary("train", train_result)

        val_result, batches = val_epoch(model=model, generator=gen_val)

        evaluation_summary("val", val_result)
        checkpointer.update_best_and_last_checkpoint(model=model, val_result=val_result)

        if isinstance(gen_train, ImageGenerator):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(
                    5,
                    len(batches),
                    name=f"epoch-{epoch:04d}",
                ),
            )
        else:
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                plot_ar_mode=experiment.misc.plot_ar_mode,
                num_ar_samples=20,
                name=f"epoch-{epoch:04d}",
            )


if __name__ == "__main__":
    main()
