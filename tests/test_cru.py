import pytest
from tqdm.auto import tqdm

from icicl.data.cru import CRUDataGenerator


def test_cru():
    # Dataset generator params.
    min_prop_ctx = 0.01
    max_prop_ctx = 0.3
    batch_grid_size = [10, 20, 20]
    lat_range = [35.25, 59.75]
    lon_range = [10.25, 44.75]
    samples_per_epoch = 16_000
    batch_size = 16
    data_dir = "/Users/matt/Downloads/dataset-derived-near-surface-meteorological-variables-3c127e0b-f66c-442a-a585-867f13ae3929/"
    fnames = ["Tair_WFDE5_CRU_201901_v2.1.nc"]

    generator = CRUDataGenerator(
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        data_dir=data_dir,
        fnames=fnames,
        min_prop_ctx=min_prop_ctx,
        max_prop_ctx=max_prop_ctx,
        batch_grid_size=batch_grid_size,
        lat_range=lat_range,
        lon_range=lon_range,
        max_nt=2000,
        min_num_total=100,
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
    test_cru()
