from utils import initialize_experiment, np_loss_fn, train_epoch, val_epoch


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

        val_epoch(model=model, generator=gen_val, epoch=epoch)


if __name__ == "__main__":
    main()
