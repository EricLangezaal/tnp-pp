import os
from abc import ABC
from typing import Optional

import numpy as np
import torch
import torchvision
from lightning.pytorch import seed_everything

from .image import ICImageGenerator, ImageGenerator


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


class ZeroShotMultiMNISTDataset(torch.utils.data.Dataset):
    """ZeroShotMultiMNIST dataset. The test set consists of multiple digits (by default 2).
    The training set consists of mnist digits with added black borders such that the image
    size is the same as in the test set, but the digits are of the same scale.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    split : {'train', 'test'}, optional
        According dataset is selected.

    n_test_digits : int, optional
        Number of digits per test image.

    final_size : int, optional
        Final size of the images (square of that shape). If `None` uses `n_test_digits*2`.

    seed : int, optional

    logger : logging.Logger

    kwargs:
        Additional arguments to the dataset data generation process `make_multi_mnist_*`.
    """

    n_classes = 0
    shape = (1, 56, 56)
    files = {"train": "train", "test": "test"}
    name = "ZeroShotMultiMNIST"

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        num_test_digits: int = 2,
        final_size: Optional[int] = None,
        translation: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        self.translation = translation
        self.seed = seed

        if train:
            if self.translation:
                transforms_list = [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.RandomCrop(
                        (self.shape[1], self.shape[2]), padding=self.translation
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            else:
                transforms_list = [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
            ]

        self.data_dir = os.path.join(data_dir, self.name)
        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.num_test_digits = num_test_digits
        self._init_size = 28

        split = "train" if train else "test"

        saved_data = os.path.join(
            self.data_dir,
            f"{split}_seed{seed}_digits{num_test_digits}.pt",
        )

        try:
            self.data = torch.load(saved_data)
        except FileNotFoundError:
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)

            mnist = torchvision.datasets.MNIST(
                root=data_dir, train=train, download=download
            )
            if train:
                self.data = self.make_multi_mnist_train(mnist.data, **kwargs)
            else:
                self.data = self.make_multi_mnist_test(mnist.data, **kwargs)

            torch.save(self.data, saved_data)

        self.data = self.data.float() / 255
        if final_size is not None:
            self.data = torch.nn.functional.interpolate(
                self.data.unsqueeze(1).float(),
                size=final_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

        self.final_size = self.data.shape[-1]

    def __len__(self):
        return self.data.size(0)

    def make_multi_mnist_train(self, train_dataset: torch.Tensor) -> torch.Tensor:
        """Train set of multi mnist by taking mnist and adding borders to be the correct scale."""
        seed_everything(self.seed)

        # Final and initial image size.
        fin_img_size = self._init_size * self.num_test_digits
        init_img_size = train_dataset.shape[1:]

        # Background is just zeros.
        background = np.zeros(
            (train_dataset.size(0), fin_img_size, fin_img_size)
        ).astype(np.uint8)

        # Who knows.
        borders = (np.array((fin_img_size, fin_img_size)) - init_img_size) // 2
        background[
            :, borders[0] : -borders[0], borders[1] : -borders[1]
        ] = train_dataset

        return torch.from_numpy(background)

    def make_multi_mnist_test(
        self,
        test_dataset: torch.Tensor,
        varying_axis: Optional[int] = None,
        num_test_digits: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Test set of multi mnist by concatenating moving digits around `varying_axis`
        (both axis if `None`) and concatenating them over the other. `n_test_digits` is th enumber
        of digits per test image (default `self.n_test_digits`).
        """
        seed_everything(self.seed)

        num_test = test_dataset.size(0)

        if num_test_digits is None:
            num_test_digits = self.num_test_digits

        if varying_axis is None:
            out_axis0 = self.make_multi_mnist_test(
                test_dataset[: num_test // 2],
                varying_axis=0,
                num_test_digits=num_test_digits,
            )
            out_axis1 = self.make_multi_mnist_test(
                test_dataset[: num_test // 2],
                varying_axis=1,
                num_test_digits=num_test_digits,
            )
            return torch.cat((out_axis0, out_axis1), dim=0)[torch.randperm(num_test)]

        fin_img_size = self._init_size * self.num_test_digits
        n_tmp = self.num_test_digits * num_test
        init_img_size = test_dataset.shape[1:]

        tmp_img_size = list(test_dataset.shape[1:])
        tmp_img_size[varying_axis] = fin_img_size
        tmp_background = torch.from_numpy(
            np.zeros((n_tmp, *tmp_img_size)).astype(np.uint8)
        )

        max_shift = fin_img_size - init_img_size[varying_axis]
        shifts = np.random.randint(max_shift, size=num_test_digits * num_test)

        test_dataset = test_dataset.repeat(self.num_test_digits, 1, 1)[
            torch.randperm(n_tmp)
        ]

        for i, shift in enumerate(shifts):
            slices = [slice(None), slice(None)]
            slices[varying_axis] = slice(shift, shift + self._init_size)
            tmp_background[i, slices[0], slices[1]] = test_dataset[i, ...]

        out = torch.cat(tmp_background.split(num_test, 0), dim=1 + 1 - varying_axis)
        return out

    def __getitem__(self, idx: int):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(self.data[idx]).float()

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class ZeroShotMultiMNISTGenerator(ImageGenerator):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        download: bool = True,
        num_test_digits: int = 2,
        final_size: Optional[int] = None,
        translation: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        self.dataset = ZeroShotMultiMNISTDataset(
            data_dir=data_dir,
            train=train,
            download=download,
            num_test_digits=num_test_digits,
            final_size=final_size,
            translation=translation,
            seed=seed,
        )
        self.dim = self.dataset.final_size * self.dataset.final_size
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)
