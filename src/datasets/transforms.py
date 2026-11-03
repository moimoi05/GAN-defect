"""Pure PyTorch and PIL transforms for Defect-GAN.

Avoids external binary dependencies (e.g. torchvision C++ extensions)
ensuring 100% portability across Windows local development and Linux Docker.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image
import torch

from src.utils import denormalize, tensor_to_rgb_numpy


class DefectGANTransforms:
    """Transforms for normal and defect patches in Defect-GAN.

    Converts PIL RGB images to 3-channel PyTorch tensors normalized to [-1, 1].
    Training split applies conservative morphology-preserving spatial flips.
    Validation and testing splits are strictly deterministic.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: Optional[int] = None,
    ) -> None:
        self.split = split
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply transforms to PIL image and return normalized tensor [3, H, W]."""
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Optional resize (primarily for smoke tests / low-VRAM debug)
        if self.image_size is not None and self.image_size != 1024:
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # Conservative morphology-preserving augmentations for training
        if self.split == "train":
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # Convert to numpy array [H, W, 3] in [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Permute to PyTorch format [3, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # Normalize from [0, 1] to [-1, 1] for Tanh activation
        tensor = (tensor - 0.5) / 0.5

        return tensor


def get_transforms(split: str = "train", image_size: Optional[int] = None) -> DefectGANTransforms:
    """Factory function for DefectGANTransforms."""
    return DefectGANTransforms(split=split, image_size=image_size)
