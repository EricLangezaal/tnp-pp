from abc import ABC
from typing import Optional, Tuple

import torchvision

from .image import ICImageGenerator, ImageGenerator
from .image_datasets import TranslationImageDataset, ZeroShotMultiImageDataset


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
        train_image_size: Optional[Tuple[int, int]] = None,
        seed: int = 0,
        **kwargs,
    ):
        mnist_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=download
        )
        self.dataset = ZeroShotMultiImageDataset(
            dataset=mnist_dataset,
            train=train,
            num_test_images=num_test_images,
            train_image_size=train_image_size,
            seed=seed,
        )
        self.dim = self.dataset.data.shape[1] * self.dataset.data.shape[2]
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)


class TranslatedMNISTGenerator(ImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        max_translation: Tuple[int, int] = (14, 14),
        stationary_image_size: Optional[Tuple[int, int]] = None,
        translated_image_size: Optional[Tuple[int, int]] = None,
        zero_shot: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        mnist_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=train, download=download
        )
        self.dataset = TranslationImageDataset(
            dataset=mnist_dataset,
            max_translation=max_translation,
            stationary_image_size=stationary_image_size,
            translated_image_size=translated_image_size,
            train=train,
            zero_shot=zero_shot,
            seed=seed,
        )
        self.dim = self.dataset.data.shape[1] * self.dataset.data.shape[2]
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)
