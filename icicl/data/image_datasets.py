from typing import Optional, Tuple

import numpy as np
import torch
import torchvision

from .image import ImageGenerator


class TranslationImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_translation: Tuple[int, int],
        stationary_image_size: Optional[Tuple[int, int, int]] = None,
        translated_image_size: Optional[Tuple[int, int, int]] = None,
        train: bool = True,
        zero_shot: bool = True,
        seed: int = 0,
    ):
        self.seed = seed
        self.image_size = dataset.data.shape[1:]
        self.max_translation = max_translation

        self.stationary_image_size = (
            self.image_size if stationary_image_size is None else stationary_image_size
        )
        self.translated_image_size = (
            [dim + 2 * shift for dim, shift in zip(self.image_size, max_translation)]
            + [self.image_size.shape[-1]]
            if translated_image_size is None
            else translated_image_size
        )

        # Make transforms.
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]
        )

        if train and zero_shot:
            self.data = self.make_stationary_images(dataset.data)
        else:
            self.data = self.make_translated_images(dataset.data)

        # Rescale between to [0, 1].
        self.data = self.data.float() / self.data.float().max()

    def make_stationary_images(self, dataset: torch.Tensor) -> torch.Tensor:
        background = np.zeros((dataset.shape[0], *self.stationary_image_size)).astype(
            np.uint8
        )

        borders = (
            np.array(self.stationary_image_size) - np.array(self.image_size)
        ) // 2
        background[
            :,
            borders[0] : (background.shape[1] - borders[0]),
            borders[1] : (background.shape[2] - borders[1]),
        ] = dataset
        return torch.from_numpy(background)

    def make_translated_images(self, dataset: torch.Tensor) -> torch.Tensor:
        background = torch.from_numpy(
            (np.zeros((dataset.shape[0], *self.translated_image_size)).astype(np.uint8))
        )

        st = np.random.get_state()
        np.random.seed(self.seed)
        vertical_shifts = np.random.randint(
            low=-self.max_translation[0],
            high=self.max_translation[0],
            size=dataset.shape[0],
        )
        horizontal_shifts = np.random.randint(
            low=-self.max_translation[1],
            high=self.max_translation[1],
            size=dataset.shape[0],
        )
        np.random.set_state(st)
        borders = (
            np.array(self.translated_image_size) - np.array(self.image_size)
        ) // 2

        for i, (vshift, hshift) in enumerate(zip(vertical_shifts, horizontal_shifts)):
            img = dataset[i, ...]

            # Trim original image to fit within background.
            if vshift < -borders[0]:
                # Trim bottom.
                img = img[-(vshift + borders[0]) :, :]
            elif vshift > borders[0]:
                # Trim top.
                img = img[: -(vshift + borders[0] - self.image_size[0]), :]

            if hshift + borders[1] < 0:
                # Trim left.
                img = img[:, -(hshift + borders[1]) :]
            elif hshift > borders[1]:
                # Trim right.
                img = img[:, : -(hshift + borders[1] - self.image_size[1])]

            vslice = slice(
                max(0, vshift + borders[0]), max(0, vshift + borders[1]) + img.shape[0]
            )
            hslice = slice(
                max(0, hshift + borders[1]), max(0, hshift + borders[1]) + img.shape[1]
            )
            background[i, vslice, hslice] = torch.as_tensor(img)

        return background

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx: int):
        # Move channel dim to first dimension.
        img = self.transforms(self.data[idx].permute(2, 0, 1)).float()
        return img, 0


class TranslatedImageGenerator(ImageGenerator):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        max_translation: Tuple[int, int] = (14, 14),
        stationary_image_size: Optional[Tuple[int, int, int]] = None,
        translated_image_size: Optional[Tuple[int, int, int]] = None,
        zero_shot: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        self.dataset = TranslationImageDataset(
            dataset=dataset,
            max_translation=max_translation,
            stationary_image_size=stationary_image_size,
            translated_image_size=translated_image_size,
            train=train,
            zero_shot=zero_shot,
            seed=seed,
        )
        self.dim = self.dataset.data.shape[1] * self.dataset.data.shape[2]
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)


class ZeroShotMultiImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        num_test_images: int = 2,
        train_image_size: Optional[Tuple[int, int]] = None,
        seed: int = 0,
        **kwargs,
    ):
        self.seed = seed
        self.num_test_images = num_test_images
        self.init_size = dataset.data.shape[1:]

        # Size of training / test images.
        self.train_image_size = (
            self.init_size if train_image_size is None else train_image_size
        )
        self.test_image_size = [size * num_test_images for size in self.init_size]

        # Make transforms.
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.num_test_images = num_test_images

        if train:
            self.data = self.make_train_images(dataset.data, **kwargs)
        else:
            self.data = self.make_test_images(dataset.data, **kwargs)

        # Rescale between to [0, 1].
        self.data = self.data.float() / self.data.float().max()

    def __len__(self):
        return self.data.size(0)

    def make_train_images(self, train_dataset: torch.Tensor) -> torch.Tensor:
        # Final and initial image sizes.
        # final_image_size = [dim * self.num_test_images for dim in self.init_size]
        final_image_size = self.train_image_size

        # Background is zeros.
        background = np.zeros((train_dataset.shape[0], *final_image_size)).astype(
            np.uint8
        )

        # Put the image in the middle of the background.
        borders = (np.array(final_image_size) - np.array(self.init_size)) // 2
        background[
            :,
            borders[0] : (background.shape[1] - borders[0]),
            borders[1] : (background.shape[2] - borders[1]),
        ] = train_dataset

        return torch.from_numpy(background)

    def make_test_images(
        self,
        test_dataset: torch.Tensor,
        varying_axis: Optional[int] = None,
    ) -> torch.Tensor:
        num_test = test_dataset.shape[0]

        if varying_axis is None:
            out_axis0 = self.make_test_images(
                test_dataset[: num_test // 2],
                varying_axis=0,
            )
            out_axis1 = self.make_test_images(
                test_dataset[: num_test // 2],
                varying_axis=1,
            )
            return torch.cat((out_axis0, out_axis1), dim=0)[torch.randperm(num_test)]

        # Final and temporary image sizes.
        # final_dim = self.init_size[varying_axis] * self.num_test_images
        final_dim = self.test_image_size[varying_axis]
        tmp_image_size = list(self.init_size)
        tmp_image_size[varying_axis] = final_dim

        # Number of temporary images.
        num_tmp = self.num_test_images * num_test

        # Backgrounds.
        tmp_background = torch.from_numpy(
            np.zeros((num_tmp, *tmp_image_size)).astype(np.uint8)
        )

        # TODO: do we not want this to be final_dim - borders[varying_axis]?
        max_shift = final_dim - self.init_size[varying_axis]

        # Set seed and restore.
        st = np.random.get_state()
        np.random.seed(self.seed)
        shifts = np.random.randint(max_shift, size=num_tmp)
        np.random.set_state(st)

        seed = torch.seed()
        torch.manual_seed(self.seed)
        test_dataset = test_dataset.repeat(self.num_test_images, 1, 1)[
            torch.randperm(num_tmp)
        ]
        torch.manual_seed(seed)

        for i, shift in enumerate(shifts):
            slices = [slice(None), slice(None)]
            slices[varying_axis] = slice(shift, shift + self.init_size[varying_axis])

            # Insert image at random shift.
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


class ZeroShotMultiImageGenerator(ImageGenerator):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        num_test_images: int = 2,
        train_image_size: Optional[Tuple[int, int]] = None,
        seed: int = 0,
        **kwargs,
    ):
        self.dataset = ZeroShotMultiImageDataset(
            dataset=dataset,
            train=train,
            num_test_images=num_test_images,
            train_image_size=train_image_size,
            seed=seed,
        )
        self.dim = self.dataset.data.shape[1] * self.dataset.data.shape[2]
        super().__init__(dataset=self.dataset, dim=self.dim, **kwargs)
