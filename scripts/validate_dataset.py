"""
Dataset Validation and Manifest Generation for Defect-GAN Baseline (114 patches).

Validates:
1. Exact approved 114 patches (47 defect, 67 clean).
2. Train (31 defect + 41 clean = 72), Val (8 defect + 9 clean = 17), Test (8 defect + 17 clean = 25).
3. RGB 3 channels and native 1024x1024 dimensions.
4. Exclusions: P25, P29, P43 (HOLD), 16 edge-clipped patches (DROP), P45 (high near-black).
5. Clean replacements used from reports/final_patch_review_v2/candidate_patches_rgb/.
6. Zero data leakage across splits at the source-image level.
"""

import csv
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Excluded patch names as mandated by final review
HOLD_PATCHES = {
    "2026-09-10-09-23-11_crop001_x4251_y610.png",  # P25 (validation)
    "2026-09-10-11-32-48_crop001_x279_y206.png",   # P29 (test)
    "2026-09-10-16-10-41_crop003_x0_y883.png",     # P43 (train)
}

EDGE_DROPPED_PATCHES = {
    "2026-09-07-22-22-53_crop001_x3833_y520.png",  # P5
    "2026-09-07-22-22-53_crop002_x2983_y0.png",    # P6
    "2026-09-07-22-22-53_crop003_x2959_y342.png",  # P7
    "2026-09-07-22-22-53_crop004_x4782_y1020.png", # P8
    "2026-09-07-22-22-53_crop005_x2651_y0.png",    # P9
    "2026-09-07-22-22-53_crop006_x2620_y0.png",    # P10
    "2026-09-07-22-22-53_crop007_x3862_y481.png",  # P11
    "2026-09-07-22-22-53_crop009_x2600_y0.png",    # P13
    "2026-09-08-09-59-55_crop001_x3542_y431.png",  # P16
    "2026-09-08-09-59-55_crop002_x2934_y320.png",  # P17
    "2026-09-08-09-59-55_crop003_x2057_y0.png",    # P18
    "2026-09-08-09-59-55_crop005_x3629_y128.png",  # P20
    "2026-09-10-15-51-23_crop001_x892_y320.png",   # P30
    "2026-09-10-16-10-51_crop002_x3029_y175.png",  # P46
    "2026-09-10-16-10-51_crop003_x1885_y799.png",  # P47
    "2026-09-10-16-10-51_crop004_x2256_y256.png",  # P48
}

HIGHLIGHTED_NG_DROPPED = {
    "2026-09-10-16-10-51_crop001_x3482_y777.png",  # P45
}

ALL_EXCLUDED_DEFECT = HOLD_PATCHES | EDGE_DROPPED_PATCHES | HIGHLIGHTED_NG_DROPPED


def load_source_splits(root: Path) -> dict:
    """Load source image to split mapping."""
    split_csv = root / "reports/source_image_splits.csv"
    mapping = {}
    with open(split_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["source_image"]] = row["split"]
    return mapping


def get_approved_clean_patches(root: Path) -> list:
    """Load approved 67 clean patches from clean_final_manifest.csv."""
    clean_csv = root / "reports/final_patch_review_v2/clean_final_manifest.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(f"Missing {clean_csv}")

    items = []
    with open(clean_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            disp = row["disposition"]
            final_rgb = row["final_rgb_file"]
            src_img = row["source_image"]
            orig_patch_id = row["original_patch_id"]

            if disp == "KEEP_ORIGINAL":
                full_path = root / "experiments/patches_1024x1024" / final_rgb
            elif disp == "REPLACE_FROM_SOURCE_RGB":
                full_path = root / "reports/final_patch_review_v2" / final_rgb
            else:
                raise ValueError(f"Unknown clean disposition: {disp}")

            if not full_path.exists():
                raise FileNotFoundError(f"Clean patch file does not exist: {full_path}")

            rel_path = str(full_path.relative_to(root)).replace("\\", "/")
            items.append({
                "patch_id": orig_patch_id,
                "filename": full_path.name,
                "domain": "clean",
                "split": split,
                "source_image": src_img,
                "patch_path": rel_path,
                "full_path": full_path,
                "disposition": disp,
            })
    return items


def get_approved_defect_patches(root: Path, source_splits: dict) -> list:
    """Load approved 47 defect patches from experiments/patches_1024x1024."""
    items = []
    splits = ["train", "validation", "test"]

    for split in splits:
        defect_dir = root / "experiments/patches_1024x1024" / split / "defect"
        if not defect_dir.exists():
            raise FileNotFoundError(f"Missing defect directory: {defect_dir}")

        for p in sorted(defect_dir.glob("*.png")):
            name = p.name
            if name in ALL_EXCLUDED_DEFECT:
                continue

            # Infer source image from filename (starts with YYYY-MM-DD-HH-MM-SS)
            stem = name.split("_crop")[0]
            src_img = f"{stem}.jpg"

            # Verify against source splits
            if src_img in source_splits:
                assigned_split = source_splits[src_img]
                if assigned_split != split:
                    raise ValueError(
                        f"Split mismatch for {name}: dir={split}, assigned={assigned_split}"
                    )

            rel_path = str(p.relative_to(root)).replace("\\", "/")
            items.append({
                "patch_id": name,
                "filename": name,
                "domain": "defect",
                "split": split,
                "source_image": src_img,
                "patch_path": rel_path,
                "full_path": p,
                "disposition": "KEEP_APPROVED",
            })
    return items


def validate_image_properties(items: list):
    """Verify all images are 1024x1024, RGB, 3 channels, non-corrupt."""
    print("Validating image resolution and channels...")
    for idx, item in enumerate(items):
        path = item["full_path"]
        with Image.open(path) as img:
            if img.mode != "RGB":
                raise ValueError(f"Image {path} mode is {img.mode}, expected RGB!")
            if img.size != (1024, 1024):
                raise ValueError(f"Image {path} size is {img.size}, expected (1024, 1024)!")
    print(f"  All {len(items)} images passed: RGB, 3-channel, 1024x1024 lossless PNG.")


def check_data_leakage(items: list):
    """Verify no source image appears in more than one split."""
    source_to_splits = {}
    for item in items:
        src = item["source_image"]
        split = item["split"]
        if src not in source_to_splits:
            source_to_splits[src] = set()
        source_to_splits[src].add(split)

    leakage_count = 0
    for src, splits in source_to_splits.items():
        if len(splits) > 1:
            print(f"DATA LEAKAGE DETECTED: {src} appears in splits: {splits}")
            leakage_count += 1

    if leakage_count > 0:
        raise RuntimeError(f"Data leakage detected in {leakage_count} source images!")
    print(f"  Source-level data leakage check: PASS ({len(source_to_splits)} unique source images).")


def write_manifests(items: list, root: Path):
    """Write train.csv, val.csv, test.csv to training/manifests/."""
    manifest_dir = root / "training" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["filename", "domain", "split", "source_image", "patch_path"]

    split_map = {
        "train": manifest_dir / "train.csv",
        "validation": manifest_dir / "val.csv",
        "test": manifest_dir / "test.csv",
    }

    split_items = {"train": [], "validation": [], "test": []}
    for item in items:
        split_items[item["split"]].append(item)

    for split_name, out_path in split_map.items():
        rows = split_items[split_name]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({
                    "filename": r["filename"],
                    "domain": r["domain"],
                    "split": r["split"],
                    "source_image": r["source_image"],
                    "patch_path": r["patch_path"],
                })
        print(f"  Wrote {len(rows)} entries to {out_path.relative_to(root)}")


def main():
    print("=" * 60)
    print("DEFECT-GAN DATASET VALIDATION & MANIFEST GENERATOR")
    print("=" * 60)

    source_splits = load_source_splits(PROJECT_ROOT)
    clean_items = get_approved_clean_patches(PROJECT_ROOT)
    defect_items = get_approved_defect_patches(PROJECT_ROOT, source_splits)

    all_items = clean_items + defect_items

    print(f"Loaded {len(clean_items)} approved clean patches.")
    print(f"Loaded {len(defect_items)} approved defect patches.")
    print(f"Total approved patches: {len(all_items)}")

    # Verify counts
    counts = {
        "train": {"defect": 0, "clean": 0},
        "validation": {"defect": 0, "clean": 0},
        "test": {"defect": 0, "clean": 0},
    }
    for item in all_items:
        counts[item["split"]][item["domain"]] += 1

    print("\nPatch Counts by Split:")
    print(f"  Train:      Defect={counts['train']['defect']} (target 31), Clean={counts['train']['clean']} (target 41), Total={counts['train']['defect'] + counts['train']['clean']} (target 72)")
    print(f"  Validation: Defect={counts['validation']['defect']} (target 8), Clean={counts['validation']['clean']} (target 9), Total={counts['validation']['defect'] + counts['validation']['clean']} (target 17)")
    print(f"  Test:       Defect={counts['test']['defect']} (target 8), Clean={counts['test']['clean']} (target 17), Total={counts['test']['defect'] + counts['test']['clean']} (target 25)")
    print(f"  GRAND TOTAL: Defect={len(defect_items)} (target 47), Clean={len(clean_items)} (target 67), Total={len(all_items)} (target 114)")

    # Assert exact targets
    assert counts["train"]["defect"] == 31, f"Expected 31 train defects, got {counts['train']['defect']}"
    assert counts["train"]["clean"] == 41, f"Expected 41 train clean, got {counts['train']['clean']}"
    assert counts["validation"]["defect"] == 8, f"Expected 8 val defects, got {counts['validation']['defect']}"
    assert counts["validation"]["clean"] == 9, f"Expected 9 val clean, got {counts['validation']['clean']}"
    assert counts["test"]["defect"] == 8, f"Expected 8 test defects, got {counts['test']['defect']}"
    assert counts["test"]["clean"] == 17, f"Expected 17 test clean, got {counts['test']['clean']}"
    assert len(all_items) == 114, f"Expected 114 total patches, got {len(all_items)}"

    # Validate image properties
    validate_image_properties(all_items)

    # Validate data leakage
    check_data_leakage(all_items)

    # Write manifests
    print("\nWriting training manifests...")
    write_manifests(all_items, PROJECT_ROOT)

    print("\nDataset validation and manifest generation COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
