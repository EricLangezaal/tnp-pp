import os
from typing import Optional

import numpy as np
import torch
import torchvision


class ZeroShotMultiImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset: torch.utils.data.Dataset,
        train: bool = True,
        num_test_images: int = 2,
        final_size: Optional[int] = None,
        translation: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        self.translation = translation
        self.seed = seed
        self.init_size = dataset.data.shape[1:]

        # Make transforms.
        if train:
            if self.translation:
                transforms_list = [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.RandomCrop(
                        (self.init_size[1], self.init_size[2]), padding=self.translation
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

        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.num_test_images = num_test_images

        # Where to save the data.
        split = "train" if train else "test"
        saved_data_fname = os.path.join(
            data_dir,
            f"{split}_seed{seed}_digits{num_test_images}.pt",
        )

        try:
            self.data = torch.load(saved_data_fname)
        except FileNotFoundError:
            # Ensure directory exists.
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            if train:
                self.data = self.make_multi_image_train(dataset.data, **kwargs)
            else:
                self.data = self.make_multi_image_test(dataset.data, **kwargs)

            torch.save(self.data, saved_data_fname)

        # Rescale between to [0, 1].
        self.data = self.data.float() / self.data.float().max()
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

    def make_multi_image_train(self, train_dataset: torch.Tensor) -> torch.Tensor:
        # Final and initial image sizes.
        final_image_size = [dim * self.num_test_images for dim in self.init_size]

        # Background is zeros.
        background = np.zeros((train_dataset.shape[0], *final_image_size)).astype(
            np.uint8
        )

        # Put the image in the middle of the background.
        borders = (np.array(final_image_size) - np.array(self.init_size)) // 2
        background[
            :, borders[0] : -borders[0], borders[1] : -borders[1]
        ] = train_dataset

        return torch.from_numpy(background)

    def make_multi_image_test(
        self,
        test_dataset: torch.Tensor,
        varying_axis: Optional[int] = None,
    ) -> torch.Tensor:
        num_test = test_dataset.shape[0]

        if varying_axis is None:
            out_axis0 = self.make_multi_image_test(
                test_dataset[: num_test // 2],
                varying_axis=0,
            )
            out_axis1 = self.make_multi_image_test(
                test_dataset[: num_test // 2],
                varying_axis=1,
            )
            return torch.cat((out_axis0, out_axis1), dim=0)[torch.randperm(num_test)]

        # Final and temporary image sizes.
        final_dim = self.init_size[varying_axis] * self.num_test_images
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
        shifts = np.random.randint(max_shift, size=self.num_test_images * num_test)
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
