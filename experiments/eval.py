import lightning.pytorch as pl
import torch
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from plot_kolmogorov import plot_kolmogorov

import wandb
from icicl.data.era5 import ERA5DataGenerator
from icicl.data.image import ImageGenerator
from icicl.data.kolmogorov import KolmogorovGenerator
from icicl.utils.data import adjust_num_batches
from icicl.utils.experiment_utils import initialize_evaluation, val_epoch
from icicl.utils.lightning_utils import LitWrapper


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.name
    gen_test = experiment.generators.val
    val_loader = torch.utils.data.DataLoader(
        gen_test,
        num_workers=gen_test.num_workers,
        batch_size=None,
        worker_init_fn=adjust_num_batches,
        persistent_workers=gen_test.num_workers > 1,
    )

    model.eval()

    if experiment.misc.only_plots:
        gen_test.batch_size = 1
        gen_test.num_batches = experiment.misc.num_plots
        _, batches = val_epoch(model=model, generator=gen_test)

        eval_name = wandb.run.name + "/" + eval_name

        if isinstance(gen_test, ImageGenerator):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(6, 6),
                name=eval_name,
                subplots=experiment.misc.subplots,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
            )
        elif isinstance(gen_test, ERA5DataGenerator):
            plot_era5(
                model=model,
                batches=batches,
                y_mean=gen_test.y_mean,
                y_std=gen_test.y_std,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(12.0, 5.0),
                name=eval_name,
                subplots=experiment.misc.subplots,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
            )
        elif isinstance(gen_test, KolmogorovGenerator):
            plot_kolmogorov(
                model=model,
                batches=batches,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                figsize=(5.0, 5.0),
                name=eval_name,
                # plot_dims=(0, 1),
                # other_dim_slice=0,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
                subplots=experiment.misc.subplots,
            )
        else:
            plot(
                model=model,
                batches=batches,
                num_fig=min(experiment.misc.num_plots, len(batches)),
                name=eval_name,
                savefig=experiment.misc.savefig,
                logging=experiment.misc.logging,
            )

        return

    # Store number of parameters.
    num_params = sum(p.numel() for p in model.parameters())
    wandb.run.summary["num_params"] = num_params

    if experiment.misc.lightning_eval:
        lit_model = LitWrapper(model,  val_generator=gen_test, optimiser=False)
        trainer = pl.Trainer(devices=1)
        trainer.test(model=lit_model, dataloaders=val_loader)
        test_result = {
            k: [result[k] for result in lit_model.test_outputs]
            for k in lit_model.test_outputs[0].keys()
        }
        loglik = torch.stack(test_result["loglik"])
        test_result["mean_loglik"] = loglik.mean()
        test_result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)
       
        if "rmse" in test_result:
            test_result["rmse"] = torch.stack(test_result["rmse"]).mean()

        if "gt_loglik" in test_result:
            gt_loglik = torch.stack(test_result["gt_loglik"])
            test_result["mean_gt_loglik"] = gt_loglik.mean()
            test_result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

        batches = test_result["batch"]

    else:
        test_result, batches = val_epoch(model=model, generator=gen_test)

    wandb.run.summary[f"test/{eval_name}/loglik"] = test_result["mean_loglik"]
    wandb.run.summary[f"test/{eval_name}/std_loglik"] = test_result["std_loglik"]
    if "rmse" in test_result:
        wandb.run.summary[f"test/{eval_name}/rmse"] = test_result["rmse"]
    if "mean_gt_loglik" in test_result:
        wandb.run.summary[f"test/{eval_name}/gt_loglik"] = test_result["mean_gt_loglik"]
        wandb.run.summary[f"test/{eval_name}/std_gt_loglik"] = test_result[
            "std_gt_loglik"
        ]

    if experiment.misc.fake_train_steps is not None:
        for epoch in range(1, experiment.params.epochs + 1):
            step = epoch * experiment.misc.fake_train_steps
            for key, value in test_result.items():
                if isinstance(value, torch.Tensor):
                    wandb.log({f"val/{key.replace('mean_', '')}": value}, step=step, commit=False)
        wandb.log({}, commit=True)            

    if isinstance(gen_test, ImageGenerator):
        plot_image(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name=f"test/{eval_name}",
        )
    elif isinstance(gen_test, ERA5DataGenerator):
        plot_era5(
            model=model,
            batches=batches,
            y_mean=gen_test.y_mean,
            y_std=gen_test.y_std,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            figsize=(15.0, 5.0),
            name=eval_name,
            subplots=experiment.misc.subplots,
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
        )
    elif isinstance(gen_test, KolmogorovGenerator):
        plot_kolmogorov(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            figsize=(18.0, 5.0),
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
            subplots=experiment.misc.subplots,
        )
    else:
        plot(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name=f"test/{eval_name}",
        )


if __name__ == "__main__":
    main()
