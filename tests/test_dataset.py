"""Unit tests for Defect-GAN dataset, manifests, and transforms."""

import csv
from pathlib import Path
import pytest
import torch

from src.datasets.f971_dataset import DefectGANDataset
from src.datasets.transforms import get_transforms
from src.utils import denormalize, tensor_to_rgb_numpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_existence_and_counts():
    """Verify train, val, and test manifests exist and have exact approved counts."""
    manifest_dir = PROJECT_ROOT / "training/manifests"
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "val.csv"
    test_csv = manifest_dir / "test.csv"

    assert train_csv.exists(), "train.csv does not exist"
    assert val_csv.exists(), "val.csv does not exist"
    assert test_csv.exists(), "test.csv does not exist"

    def count_rows(p):
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))
            defect_cnt = sum(1 for r in reader if r["domain"] == "defect")
            clean_cnt = sum(1 for r in reader if r["domain"] == "clean")
            return defect_cnt, clean_cnt, len(reader)

    d_tr, c_tr, tot_tr = count_rows(train_csv)
    d_val, c_val, tot_val = count_rows(val_csv)
    d_te, c_te, tot_te = count_rows(test_csv)

    assert (d_tr, c_tr, tot_tr) == (31, 41, 72), f"Train counts mismatch: {(d_tr, c_tr, tot_tr)}"
    assert (d_val, c_val, tot_val) == (8, 9, 17), f"Val counts mismatch: {(d_val, c_val, tot_val)}"
    assert (d_te, c_te, tot_te) == (8, 17, 25), f"Test counts mismatch: {(d_te, c_te, tot_te)}"
    assert tot_tr + tot_val + tot_te == 114, "Grand total must be 114"


def test_no_source_image_leakage():
    """Verify source images are strictly partitioned across train/val/test."""
    manifest_dir = PROJECT_ROOT / "training/manifests"
    splits = ["train", "val", "test"]
    source_to_splits = {}

    for s in splits:
        with open(manifest_dir / f"{s}.csv", "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                src = r["source_image"]
                if src not in source_to_splits:
                    source_to_splits[src] = set()
                source_to_splits[src].add(s)

    leaks = {src: sp for src, sp in source_to_splits.items() if len(sp) > 1}
    assert len(leaks) == 0, f"Data leakage detected across splits: {leaks}"


def test_dataset_loading_item():
    """Verify DefectGANDataset loads RGB tensors in [-1, 1] range."""
    train_manifest = PROJECT_ROOT / "training/manifests/train.csv"
    dataset = DefectGANDataset(
        manifest_path=train_manifest,
        data_root=PROJECT_ROOT,
        split="train",
        image_size=128,  # Small size for fast test
    )

    assert len(dataset) == 41  # max(31, 41)
    sample = dataset[0]

    assert "normal" in sample
    assert "defect" in sample
    assert sample["normal"].shape == (3, 128, 128)
    assert sample["defect"].shape == (3, 128, 128)
    assert sample["normal"].dtype == torch.float32
    assert sample["defect"].dtype == torch.float32

    # Check range [-1, 1]
    assert sample["normal"].min() >= -1.05 and sample["normal"].max() <= 1.05
    assert sample["defect"].min() >= -1.05 and sample["defect"].max() <= 1.05


def test_denormalize_and_rgb_numpy():
    """Verify denormalize and RGB conversion utilities."""
    tensor = torch.full((3, 64, 64), 0.0)  # midpoint in [-1, 1] -> 0.5 in [0, 1]
    denorm = denormalize(tensor)
    assert torch.allclose(denorm, torch.full((3, 64, 64), 0.5))

    np_img = tensor_to_rgb_numpy(tensor)
    assert np_img.shape == (64, 64, 3)
    assert np_img.dtype == "uint8"
    assert np_img[0, 0, 0] in (127, 128)
