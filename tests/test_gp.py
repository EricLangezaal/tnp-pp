import pytest
from tqdm.auto import tqdm

from icicl.data.gp import RandomScaleGPGenerator


@pytest.mark.parametrize(
    "kernel",
    # ["eq", "matern12", "matern32", "matern52", "noisy_mixture", "weakly_periodic"],
    ["eq"],
)
def test_gp(kernel: str):
    # GP params.
    min_log10_lengthscale = -0.607
    max_log10_lengthscale = 0.607
    dim = 1
    min_num_ctx = 1
    max_num_ctx = 512
    min_num_trg = 512
    max_num_trg = 512
    context_range = ((-2.0, 2.0),)
    target_range = ((-3.0, 3.0),)

    samples_per_epoch = 2
    batch_size = 3

    generator = RandomScaleGPGenerator(
        kernel_type=kernel,
        min_log10_lengthscale=min_log10_lengthscale,
        max_log10_lengthscale=max_log10_lengthscale,
        noise_std=0.1,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        dim=dim,
        min_num_ctx=min_num_ctx,
        max_num_ctx=max_num_ctx,
        min_num_trg=min_num_trg,
        max_num_trg=max_num_trg,
        context_range=context_range,
        target_range=target_range,
    )

    # Training params.
    epochs = 10

    batches = []
    for _ in range(epochs):
        epoch = tqdm(generator, total=generator.num_batches, desc="Training")

        for batch in epoch:
            batches.append(batch)

    assert len(batches) == (epochs * generator.num_batches)
