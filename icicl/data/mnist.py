from abc import ABC

import torchvision

from .image import ImageGenerator
from .image_datasets import TranslatedImageGenerator, ZeroShotMultiImageGenerator


class MNIST(ABC):
    def __init__(self, data_dir: str, train: bool = True, download: bool = False):
        self.dim = 28 * 28
        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )


class MNISTGenerator(MNIST, ImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        MNIST.__init__(self, data_dir, train, download)
        # Add channel dimension.
        self.dataset.data = self.dataset.data.unsqueeze(-1)
        # Rescale between to [0, 1].
        self.dataset.data = self.dataset.data.float() / self.dataset.data.float().max()
        ImageGenerator.__init__(self, dataset=self.dataset, dim=self.dim, **kwargs)


class ZeroShotMultiMNISTGenerator(ZeroShotMultiImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        mnist_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=download
        )
        mnist_dataset.data = mnist_dataset.data.unsqueeze(-1)
        super().__init__(dataset=mnist_dataset, train=train, **kwargs)


class TranslatedMNISTGenerator(TranslatedImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        mnist_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=download
        )
        # Add channel dimension to MNIST.
        mnist_dataset.data = mnist_dataset.data.unsqueeze(-1)
        super().__init__(dataset=mnist_dataset, train=train, **kwargs)
