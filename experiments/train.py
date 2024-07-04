import torch

from plot import plot
from experiments.plot_globe import plot_globe
from plot_image import plot_image
from plot_kolmogorov import plot_kolmogorov

from icicl.utils.data import adjust_num_batches
from icicl.data.era5 import ERA5DataGenerator
from icicl.data.cru import CRUDataGenerator
from icicl.data.image import ImageGenerator
from icicl.data.kolmogorov import KolmogorovGenerator
from icicl.utils.experiment_utils import (
    evaluation_summary,
    initialize_experiment,
    train_epoch,
    val_epoch,
)


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        num_workers=experiment.misc.num_workers,
        batch_size=None,
        worker_init_fn=adjust_num_batches,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        num_workers=experiment.misc.num_workers,
        batch_size=None,
        worker_init_fn=adjust_num_batches,
    )

    step = 0
    for epoch in range(epochs):
        model.train()
        step, train_result = train_epoch(
            model=model,
            generator=train_loader,
            optimiser=optimiser,
            step=step,
            loss_fn=experiment.misc.loss_fn,
            gradient_clip_val=experiment.misc.gradient_clip_val,
        )
        model.eval()
        evaluation_summary("train", train_result)
        checkpointer.update_best_and_last_checkpoint(
            model=model, val_result=train_result, prefix="train_", update_last=True
        )

        val_result, batches = val_epoch(model=model, generator=val_loader)

        evaluation_summary("val", val_result)
        checkpointer.update_best_and_last_checkpoint(
            model=model, val_result=val_result, prefix="val_", update_last=False
        )

        if epoch % experiment.misc.plot_interval == 0:
            if isinstance(gen_train, ImageGenerator):
                plot_image(
                    model=model,
                    batches=batches,
                    num_fig=min(5, len(batches)),
                    name=f"epoch-{epoch:04d}",
                )
            elif isinstance(gen_train, CRUDataGenerator) or isinstance(gen_train, ERA5DataGenerator):
                plot_globe(
                    model=model,
                    batches=batches,
                    x_mean=gen_val.x_mean if isinstance(gen_train, CRUDataGenerator) else None,
                    x_std=gen_val.x_std if isinstance(gen_train, CRUDataGenerator) else None,
                    y_mean=gen_val.y_mean,
                    y_std=gen_val.y_std,
                    num_fig=min(5, len(batches)),
                    figsize=(24.0, 5.0),
                    lat_range=gen_val.lat_range,
                    lon_range=gen_val.lon_range,
                    time_idx=(0, -1) if isinstance(gen_train, CRUDataGenerator) else None,
                    name=f"epoch-{epoch:04d}",
                )
            elif isinstance(gen_train, KolmogorovGenerator):
                plot_kolmogorov(
                    model=model,
                    batches=batches,
                    num_fig=min(5, len(batches)),
                    figsize=(18.0, 5.0),
                    savefig=False,
                    logging=False,
                    subplots=False,
                )
            else:
                plot(
                    model=model,
                    batches=batches,
                    num_fig=min(5, len(batches)),
                    name=f"epoch-{epoch:04d}",
                )


if __name__ == "__main__":
    main()
