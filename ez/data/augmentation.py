# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


from kornia.augmentation import RandomAffine, RandomCrop, CenterCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Transforms(object):
    """ Reference : Data-Efficient Reinforcement Learning with Self-Predictive Representations
    Thanks to Repo: https://github.com/mila-iqia/spr.git
    """
    def __init__(self, augmentation, shift_delta=4, image_shape=(96, 96)):
        self.augmentation = augmentation

        self.transforms = []
        for aug in self.augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
            elif aug == "crop":
                transformation = RandomCrop(image_shape)
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
            elif aug == "shift":
                # transformation = nn.Sequential(nn.ReplicationPad2d(shift_delta), RandomCrop(image_shape))
                transformation = RandomShiftsAug(pad=shift_delta)
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
            elif aug == "none":
                transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = transform(image)
        return image

    @torch.no_grad()
    def transform(self, images):
        # images = images.float() / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        processed_images = self.apply_transforms(self.transforms, flat_images)

        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    @torch.no_grad()
    def __call__(self, images):
        return self.transform(images)


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise
