from plot import plot
from utils import (
    evaluation_summary,
    initialize_experiment,
    np_loss_fn,
    train_epoch,
    val_epoch,
)


def main():
    experiment = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    step = 0
    for epoch in range(epochs):
        step = train_epoch(
            model=model,
            generator=gen_train,
            optimiser=optimiser,
            step=step,
            loss_fn=np_loss_fn,
        )

        val_result, batches = val_epoch(model=model, generator=gen_val, epoch=epoch)

        evaluation_summary("val", val_result)
        plot(model=model, batches=batches, epoch=epoch, num_fig=5)


if __name__ == "__main__":
    main()
