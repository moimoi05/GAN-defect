"""Utility functions for Defect-GAN."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Tensor with values roughly in [-1, 1].

    Returns:
        Tensor clamped to [0, 1].
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch image tensor [3, H, W] or [1, H, W] to uint8 NumPy array [H, W, 3].

    Args:
        tensor: Float tensor on CPU or GPU in range [-1, 1] or [0, 1].

    Returns:
        NumPy uint8 array of shape [H, W, 3] in RGB format.
    """
    t = tensor.detach().to(device='cpu', dtype=torch.float32)
    if t.dim() == 4:
        t = t.squeeze(0)  # Handle [1, C, H, W]

    # If 1-channel mask, repeat to 3 channels for visualization
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
        # Masks are [0, 1], don't denormalize [-1, 1]
        t = torch.clamp(t, 0.0, 1.0)
    else:
        # RGB images are [-1, 1]
        t = denormalize(t)

    np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return np_img


def save_rgb_image(tensor: torch.Tensor, output_path: Union[str, Path]) -> Path:
    """Save a PyTorch image tensor as a lossless RGB PNG.

    Args:
        tensor: Float tensor [3, H, W] in [-1, 1] (or [1, H, W] mask).
        output_path: File path to save PNG.

    Returns:
        Path to saved image.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_img = tensor_to_rgb_numpy(tensor)
    img = Image.fromarray(np_img, mode="RGB")
    img.save(path, format="PNG")
    return path


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a PyTorch module.

    Args:
        model: PyTorch module.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def check_tensor_valid(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Check that a tensor contains no NaN or Inf values.

    Args:
        tensor: Tensor to check.
        name: Name for error reporting.

    Raises:
        ValueError if NaN or Inf is found.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"Found NaN values in {name}!")
    if torch.isinf(tensor).any():
        raise ValueError(f"Found Inf values in {name}!")
