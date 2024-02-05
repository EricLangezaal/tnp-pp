import pytest
from tqdm.auto import tqdm

from icicl.data.gp import RandomScaleGPGenerator, RandomScaleGPGeneratorBimodalInput
from icicl.data.synthetic import SyntheticGeneratorMixture


def test_mixture():
    # Generator params.
    dim = 1
    min_num_ctx = 1
    max_num_ctx = 512
    min_num_trg = 512
    max_num_trg = 512
    samples_per_epoch = 16
    batch_size = 8

    # GP params.
    kernel = "eq"
    min_log10_lengthscale = -0.607
    max_log10_lengthscale = 0.607
    noise_std = 0.1

    # Unimodal params.
    context_range = ((-2.0, 2.0),)
    target_range = ((-3.0, 3.0),)

    # Bimodal params.
    bi_context_range = (((-4.0, -1.0), (1.0, 4.0)),)
    bi_target_range = (((-5.0, -1.0), (1.0, 5.0)),)

    # Mixture probs.
    mixture_probs = (0.5, 0.5)

    # Unimodal generator.
    uni_generator = RandomScaleGPGenerator(
        kernel_type=kernel,
        min_log10_lengthscale=min_log10_lengthscale,
        max_log10_lengthscale=max_log10_lengthscale,
        noise_std=noise_std,
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

    # Bimodal generator.
    bi_generator = RandomScaleGPGeneratorBimodalInput(
        kernel_type=kernel,
        min_log10_lengthscale=min_log10_lengthscale,
        max_log10_lengthscale=max_log10_lengthscale,
        noise_std=noise_std,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        dim=dim,
        min_num_ctx=min_num_ctx,
        max_num_ctx=max_num_ctx,
        min_num_trg=min_num_trg,
        max_num_trg=max_num_trg,
        context_range=bi_context_range,
        target_range=bi_target_range,
    )

    # Mixture generator.
    generator = SyntheticGeneratorMixture(
        generators=(uni_generator, bi_generator),
        mixture_probs=mixture_probs,
        min_num_ctx=min_num_ctx,
        max_num_ctx=max_num_ctx,
        min_num_trg=min_num_trg,
        max_num_trg=max_num_trg,
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        dim=dim,
    )

    epochs = 10
    batches = []
    for _ in range(epochs):
        epoch = tqdm(generator, total=generator.num_batches)

        for batch in epoch:
            batches.append(batch)


if __name__ == "__main__":
    test_mixture()
