import torch
from tqdm.auto import tqdm

from icicl.data.mnist import MNISTGenerator


def test_mnist_generator():

    epochs = 2
    batch_size = 16
    min_prop_ctx = 0.0
    max_prop_ctx = 1.0
    data_dir = "/Users/matt/projects/tnp-pp/data"
    train = True
    download = True
    samples_per_epoch = 100

    generator = MNISTGenerator(
        batch_size=batch_size,
        min_prop_ctx=min_prop_ctx,
        max_prop_ctx=max_prop_ctx,
        data_dir=data_dir,
        train=train,
        download=download,
        samples_per_epoch=samples_per_epoch,
    )

    for _ in range(epochs):
        batches = []
        epoch = tqdm(generator, total=generator.num_batches, desc="Training")

        for batch in epoch:
            batches.append(batch)

        assert len(batches) == generator.num_batches


def test_mnist_generator_same_label():

    epochs = 2
    batch_size = 16
    min_prop_ctx = 0.0
    max_prop_ctx = 1.0
    data_dir = "/Users/matt/projects/tnp-pp/data"
    train = True
    download = True
    samples_per_epoch = 200
    sample_label_per_batch = True

    generator = MNISTGenerator(
        batch_size=batch_size,
        min_prop_ctx=min_prop_ctx,
        max_prop_ctx=max_prop_ctx,
        data_dir=data_dir,
        train=train,
        download=download,
        samples_per_epoch=samples_per_epoch,
        same_label_per_batch=sample_label_per_batch,
    )

    for _ in range(epochs):
        batches = []
        epoch = tqdm(generator, total=generator.num_batches, desc="Training")

        for batch in epoch:
            assert len(batch.xc) == batch_size
            assert torch.equal(batch.label.float().std(), torch.as_tensor(0))
            batches.append(batch)

        assert len(batches) == generator.num_batches
