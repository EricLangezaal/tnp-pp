import lightning.pytorch as pl
import torch

from plot import plot
from plot_era5 import plot_era5

from tnp.data.on_off_grid import RandomOOTGGenerator
from tnp.data.era5 import ERA5DataGenerator
from tnp.utils.data import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        num_workers=gen_train.num_workers,
        batch_size=None,
        worker_init_fn=adjust_num_batches,
        persistent_workers=gen_train.num_workers > 1,
        prefetch_factor=5 if gen_train.num_workers > 1 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        num_workers=gen_val.num_workers,
        batch_size=None,
        worker_init_fn=adjust_num_batches,
        persistent_workers=gen_val.num_workers > 1,
        prefetch_factor=4 if gen_val.num_workers > 1 else None,
    )

    if isinstance(gen_val, ERA5DataGenerator):
                
        def plot_fn(model, batches, name):
                plot_era5(
                    model=model,
                    batches=batches,
                    y_mean=gen_val.y_mean,
                    y_std=gen_val.y_std,
                    num_fig=min(5, len(batches)),
                    figsize=(12.0, 5.0),
                    name=name,
                    subplots=False,
                )
    elif isinstance(gen_val, RandomOOTGGenerator):
        plot_fn = None
    else:

        def plot_fn(model, batches, name):
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
    torch.multiprocessing.set_start_method('spawn', force=True)

    lit_model = LitWrapper(
        model=model,
        optimiser=optimiser,
        val_generator=gen_val,
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
        num_sanity_val_steps=0,
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
