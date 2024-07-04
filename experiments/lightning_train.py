import lightning.pytorch as pl
import torch
from plot import plot
from experiments.plot_globe import plot_globe
from plot_image import plot_image
from plot_kolmogorov import plot_kolmogorov

from icicl.data.cru import CRUDataGenerator
from icicl.data.era5 import ERA5DataGenerator
from icicl.data.image import ImageGenerator
from icicl.data.kolmogorov import KolmogorovGenerator
from icicl.utils.data import adjust_num_batches
from icicl.utils.experiment_utils import initialize_experiment
from icicl.utils.lightning_utils import LitWrapper


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

    if isinstance(gen_val, ImageGenerator):

        def plot_fn(model, batches, name):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )

    elif isinstance(gen_val, CRUDataGenerator) or isinstance(gen_val, ERA5DataGenerator):
                
        def plot_fn(model, batches, name):
                plot_globe(
                    model=model,
                    batches=batches,
                    x_mean=gen_val.x_mean if isinstance(gen_val, CRUDataGenerator) else None,
                    x_std=gen_val.x_std if isinstance(gen_val, CRUDataGenerator) else None,
                    y_mean=gen_val.y_mean,
                    y_std=gen_val.y_std,
                    num_fig=min(5, len(batches)),
                    figsize=(24.0, 5.0),
                    lat_range=gen_val.lat_range,
                    lon_range=gen_val.lon_range,
                    time_idx=(0, -1) if isinstance(gen_val, CRUDataGenerator) else None,
                    name=name,
                )

    elif isinstance(gen_val, KolmogorovGenerator):

        def plot_fn(model, batches, name):
            plot_kolmogorov(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                figsize=(18.0, 5.0),
                subplots=True,
                name=name,
            )

    else:

        def plot_fn(model, batches, name):
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )

    lit_model = LitWrapper(
        model=model,
        optimiser=optimiser,
        loss_fn=experiment.misc.loss_fn,
        pred_fn=experiment.misc.pred_fn,
        plot_fn=plot_fn,
        checkpointer=checkpointer,
        plot_interval=experiment.misc.plot_interval,
    )
    logger = pl.loggers.WandbLogger() if experiment.misc.logging else False
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=1,
        devices=1, #more?
        # strategy="ddp",
        gradient_clip_val=experiment.misc.gradient_clip_val,
        enable_progress_bar=experiment.misc.progress_bars,
        accelerator="gpu",
    )

    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()
