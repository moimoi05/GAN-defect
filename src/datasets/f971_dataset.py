"""PyTorch Dataset for Unpaired Defect-GAN RGB Image Translation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets.transforms import get_transforms


class DefectGANDataset(Dataset):
    """Unpaired PyTorch Dataset for Normal (OK) and Defect (NG) RGB images.

    Loads images according to manifest CSV files (train.csv, val.csv, test.csv).
    Maintains separate pools for the Normal domain (clean F971) and Defect domain (NG F971).
    For training, samples from both domains independently.
    For validation/test, iterates deterministically.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        split: str = "train",
        image_size: Optional[int] = 1024,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialise DefectGANDataset.

        Args:
            manifest_path: Path to manifest CSV (train.csv, val.csv, test.csv).
            data_root: Base directory for relative paths in manifest. If None,
                       uses current working directory or repository root.
            split: 'train', 'validation', or 'test'.
            image_size: Target image size (default 1024).
            transform: Optional transform callable. If None, default transforms
                       for split are used.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else Path(".")
        self.split = split
        self.image_size = image_size
        self.transform = transform or get_transforms(split=split, image_size=image_size)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.normal_items: List[Dict[str, str]] = []
        self.defect_items: List[Dict[str, str]] = []

        self._load_manifest()

        if len(self.normal_items) == 0:
            raise ValueError(f"No clean/normal images found in {self.manifest_path}")
        if len(self.defect_items) == 0:
            raise ValueError(f"No defect images found in {self.manifest_path}")

    def _load_manifest(self) -> None:
        """Read manifest CSV and partition into normal and defect items."""
        with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row["domain"].strip().lower()
                rel_path = row["patch_path"].strip()

                # Verify file can be resolved relative to data_root
                full_path = self.data_root / rel_path
                if not full_path.exists():
                    # Fallback: check if path is relative to manifest directory's parent
                    alt_path = self.manifest_path.resolve().parent.parent.parent / rel_path
                    if alt_path.exists():
                        full_path = alt_path
                    else:
                        raise FileNotFoundError(
                            f"Image patch not found: {full_path} (also checked {alt_path})"
                        )

                item = {
                    "filename": row.get("filename", full_path.name),
                    "domain": domain,
                    "split": row.get("split", self.split),
                    "source_image": row.get("source_image", ""),
                    "patch_path": rel_path,
                    "full_path": str(full_path),
                }

                if domain == "clean" or domain == "normal":
                    self.normal_items.append(item)
                elif domain == "defect" or domain == "ng":
                    self.defect_items.append(item)
                else:
                    raise ValueError(f"Unknown domain in manifest: {domain}")

    def __len__(self) -> int:
        """Dataset length is maximum of domain counts to expose all samples."""
        return max(len(self.normal_items), len(self.defect_items))

    def _load_image(self, path: str) -> torch.Tensor:
        """Open image as RGB and apply transforms."""
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Fetch an unpaired pair of (normal, defect) RGB image tensors.

        Args:
            idx: Index in range [0, len(self)).

        Returns:
            Dict containing:
                - 'normal': Tensor [3, H, W] in [-1, 1]
                - 'defect': Tensor [3, H, W] in [-1, 1]
                - 'normal_path': Path string to normal image
                - 'defect_path': Path string to defect image
        """
        if self.split == "train":
            # For training: index normal cyclically, sample defect randomly
            # (or vice-versa) to ensure unpaired variability
            normal_idx = idx % len(self.normal_items)
            defect_idx = random.randint(0, len(self.defect_items) - 1)
        else:
            # Deterministic pairing for validation/testing
            normal_idx = idx % len(self.normal_items)
            defect_idx = idx % len(self.defect_items)

        normal_item = self.normal_items[normal_idx]
        defect_item = self.defect_items[defect_idx]

        normal_tensor = self._load_image(normal_item["full_path"])
        defect_tensor = self._load_image(defect_item["full_path"])

        return {
            "normal": normal_tensor,
            "defect": defect_tensor,
            "normal_path": normal_item["patch_path"],
            "defect_path": defect_item["patch_path"],
            "normal_source": normal_item["source_image"],
            "defect_source": defect_item["source_image"],
        }
