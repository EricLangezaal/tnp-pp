from abc import ABC

import torchvision

from .image import ImageGenerator
from .image_datasets import TranslatedImageGenerator, ZeroShotMultiImageGenerator


class CIFAR10(ABC):
    def __init__(self, data_dir: str, train: bool = True, download: bool = False):
        self.dim = 28 * 28
        self.dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            ),
        )


class CIFAR10Generator(CIFAR10, ImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        CIFAR10.__init__(self, data_dir, train, download)
        # Rescale between to [0, 1].
        self.dataset.data = self.dataset.data.float() / self.dataset.data.float().max()
        ImageGenerator.__init__(self, dataset=self.dataset, dim=self.dim, **kwargs)


class ZeroShotMultiCIFAR10Generator(ZeroShotMultiImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        cifar10_dataset = torchvision.datasets.MNCIFAR10ST(
            root=data_dir, train=train, download=download
        )
        super().__init__(dataset=cifar10_dataset, train=train, **kwargs)


class TranslatedCIFAR10Generator(TranslatedImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ):
        cifar10_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=download
        )
        super().__init__(dataset=cifar10_dataset, train=train, **kwargs)
