from abc import ABC
from typing import Optional

import torchvision

from .image import ICImageGenerator, ImageGenerator
from .image_datasets import ZeroShotMultiImageDataset


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
        ImageGenerator.__init__(self, dataset=self.dataset, dim=self.dim, **kwargs)


class ICMNISTGenerator(MNIST, ICImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        MNIST.__init__(self, data_dir, train, download)
        ICImageGenerator.__init__(self, dataset=self.dataset, dim=self.dim, **kwargs)


class ZeroShotMultiMNISTGenerator(ImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        num_test_images: int = 2,
        final_size: Optional[int] = None,
        translation: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        mnist_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=download
        )
        self.dataset = ZeroShotMultiImageDataset(
            data_dir=data_dir,
            dataset=mnist_dataset,
            train=train,
            num_test_images=num_test_images,
            final_size=final_size,
            translation=translation,
            seed=seed,
        )
        self.dim = self.dataset.final_size * self.dataset.final_size
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)
