from plot import plot
from plot_image import plot_image
from utils import evaluation_summary, initialize_evaluation, val_epoch

from icicl.data.image import ImageGenerator


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    gen_val = experiment.generators.val

    val_result, batches = val_epoch(model=model, generator=gen_val)
    evaluation_summary("test", val_result)

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
