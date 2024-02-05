import pytest
from tqdm.auto import tqdm

from icicl.data.kolmogorov import KolmogorovGenerator


def test_kolmogorov():
    # Dataset generator params.
    data_dir = "/Users/matt/Downloads/kolmogorov"
    split = "train"
    forecasting = False
    batch_grid_size = [16, 16, 16]
    min_num_ctx = 1
    max_num_ctx = 1000
    min_num_trg = 1000
    max_num_trg = 1000
    samples_per_epoch = 16000
    batch_size = 16

    generator = KolmogorovGenerator(
        data_dir=data_dir,
        split=split,
        forecasting=forecasting,
        batch_grid_size=batch_grid_size,
        min_num_ctx=min_num_ctx,
        max_num_ctx=max_num_ctx,
        min_num_trg=min_num_trg,
        max_num_trg=max_num_trg,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
    )
    # Training params.
    epochs = 1

    batches = []
    for _ in range(epochs):
        epoch = tqdm(
            generator,
            total=generator.num_batches,
            desc="Training",
        )

        for batch in epoch:
            batches.append(batch)

    assert len(batches) == (epochs * generator.num_batches)


if __name__ == "__main__":
    test_kolmogorov()
