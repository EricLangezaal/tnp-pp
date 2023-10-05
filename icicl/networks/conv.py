from typing import List

import torch
from check_shapes import check_shapes
from torch import nn

CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

TRANSPOSE_CONV = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        first_channels: int,
        last_channels: int,
        kernel_size: int,
        num_channels: List[int],
        activation: nn.Module = nn.ReLU(),
        pool: nn.Module = nn.MaxPool1d,
        pooling_size: int = 2,
        **kwargs,
    ):
        """Constructs a UNet-based convolutional architecture, consisting
        of a first convolutional layer, followed by a UNet style architecture
        with skip connections, and a final convolutional layer.

        Arguments:
            first_channels: Number of channels in the first convolutional layer.
            last_channels: Number of channels in the last convolutional layer.
            kernel_size: Size of the convolutional kernels.
            num_channels: Number of channels in each UNet layer.
            strides: Strides in each UNet layer.
            dim: Dimensionality of the input data.
            seed: Random seed.
            activation: Activation function to use.
            pooling_size: Size of the pooling.
            **kwargs: Additional keyword arguments.
        """
        assert dim in [1, 2, 3], f"UNet dim must be in [1, 2, 3], found {dim=}."

        super().__init__(**kwargs)

        self.dim = dim
        self.activation = activation
        self.pooling = pool(pooling_size)
        self.convs = []
        self.transposed_convs = []

        def shared_kwargs(i, o, k):
            return {
                "in_channels": i,
                "out_channels": o,
                "kernel_size": k,
                "padding": k // 2,
            }

        # First convolutional layer
        self.first = CONV[dim](
            **shared_kwargs(in_channels, first_channels, kernel_size)
        )

        # UNet layers
        for i, num_channel in enumerate(num_channels):
            if i == 0:
                prev_channels = first_channels
                upwards_multiplier = 1
            else:
                prev_channels = num_channels[i - 1]
                upwards_multiplier = 2

            if i == (len(num_channels) - 1):
                next_channels = first_channels
            else:
                next_channels = num_channels[-(i + 2)]

            self.convs.append(
                CONV[dim](**shared_kwargs(prev_channels, num_channel, kernel_size))
            )

            self.transposed_convs.append(
                TRANSPOSE_CONV[dim](
                    **shared_kwargs(
                        upwards_multiplier * num_channels[-(i + 1)],
                        next_channels,
                        kernel_size,
                    )
                )
            )

        # Last convolutional layer
        self.last = TRANSPOSE_CONV[dim](
            **shared_kwargs(2 * first_channels, last_channels, kernel_size)
        )

    @check_shapes("z: [m, ..., c]")
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Move channels to after batch dimension.
        z = torch.movedim(z, -1, 1)

        z = self.first(z)
        skips = []

        for conv in self.convs:
            skips.append(z)
            z = self.pooling(z)
            z = self.activation(conv(z))

        for conv, skip in zip(self.transposed_convs, skips[::-1]):
            z = nn.functional.interpolate(
                z,
                scale_factor=self.pooling.kernel_size,
                mode="linear",
                align_corners=True,
            )
            z = self.activation(conv(z))

            if z.shape[-1] < skip.shape[-1]:
                # Compute and apply padding.
                pad = [0, skip.shape[-1] - z.shape[-1]] * self.dim
                z = nn.functional.pad(z, pad, "constant", 0)

            z = torch.concat([z, skip], dim=-(self.dim + 1))

        z = self.last(z)

        # Move channels to final dimension.
        z = torch.movedim(z, 1, -1)

        return z
