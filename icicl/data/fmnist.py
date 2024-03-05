from abc import ABC

import torchvision

from .image import ImageGenerator


class FashionMNIST(ABC):
    def __init__(self, data_dir: str, train: bool = True, download: bool = False):
        self.dim = 28 * 28
        self.dataset = torchvision.datasets.FashionMNIST(
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


class FashionMNISTGenerator(FashionMNIST, ImageGenerator):
    def __init__(
        self, *, data_dir: str, train: bool = True, download: bool = False, **kwargs
    ):
        FashionMNIST.__init__(self, data_dir, train, download)
        ImageGenerator.__init__(self, dataset=self.dataset, dim=self.dim, **kwargs)
