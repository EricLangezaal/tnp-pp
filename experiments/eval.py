import lightning.pytorch as pl
import torch
from lightning_utils import LitWrapper
from plot import plot
from plot_cru import plot_cru
from plot_image import plot_image
from utils import initialize_evaluation, val_epoch

import wandb
from icicl.data.cru import CRUDataGenerator
from icicl.data.image import ImageGenerator


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    # Store number of parameters.
    num_params = sum(p.numel() for p in model.parameters())
    wandb.run.summary["num_params"] = num_params

    if experiment.misc.lightning_eval:
        lit_model = LitWrapper(model)
        trainer = pl.Trainer(devices=1)
        trainer.test(model=lit_model, dataloaders=gen_test)
        test_result = {
            k: [result[k] for result in lit_model.test_outputs]
            for k in lit_model.test_outputs[0].keys()
        }
        loglik = torch.stack(test_result["loglik"])
        test_result["mean_loglik"] = loglik.mean()
        test_result["std_loglik"] = loglik.std() / (len(loglik) ** 0.5)

        if "gt_loglik" in test_result:
            gt_loglik = torch.stack(test_result["gt_loglik"])
            test_result["mean_gt_loglik"] = gt_loglik.mean()
            test_result["std_gt_loglik"] = gt_loglik.std() / (len(gt_loglik) ** 0.5)

        batches = test_result["batch"]

    else:
        test_result, batches = val_epoch(model=model, generator=gen_test)

    wandb.run.summary[f"test/{eval_name}/loglik"] = test_result["mean_loglik"]
    wandb.run.summary[f"test/{eval_name}/std_loglik"] = test_result["std_loglik"]
    if "mean_gt_loglik" in test_result:
        wandb.run.summary[f"test/{eval_name}/gt_loglik"] = test_result["mean_gt_loglik"]
        wandb.run.summary[f"test/{eval_name}/std_gt_loglik"] = test_result[
            "std_gt_loglik"
        ]

    if isinstance(gen_test, ImageGenerator):
        plot_image(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name=f"test/{eval_name}",
        )
    elif isinstance(gen_test, CRUDataGenerator):
        plot_cru(
            model=model,
            batches=batches,
            x_mean=gen_test.x_mean,
            x_std=gen_test.x_std,
            y_mean=gen_test.y_mean,
            y_std=gen_test.y_std,
            num_fig=min(experiment.misc.numy_plots, len(batches)),
            figsize=(24.0, 5.0),
            lat_range=gen_test.lat_range,
            lon_range=gen_test.lon_range,
            time_idx=(0, -1),
            name=f"test/{eval_name}/",
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
