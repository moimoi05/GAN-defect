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

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.datasets.f971_dataset import resolve_patch_path

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
            if src_img not in source_splits:
                raise ValueError(f"Source missing from approved source split map: {src_img}")
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
            if img.format != "PNG":
                raise ValueError(f"Image {path} is not PNG")
            img.load()  # Decode pixels to detect truncated/corrupt files.
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


EXPECTED_COUNTS = {"train": {"clean": 41, "defect": 31},
                   "validation": {"clean": 9, "defect": 8},
                   "test": {"clean": 17, "defect": 8}}
FIELDS = ["filename", "domain", "split", "source_image", "patch_path"]


def approved_items(root: Path) -> list:
    """Read approved audit records; never write data or infer new patches."""
    return get_approved_clean_patches(root) + get_approved_defect_patches(root, load_source_splits(root))


def validate_baseline(data_root: Path, manifest_dir: Path = None) -> dict:
    """Validate existing manifests against approved audits, strictly read-only."""
    root = Path(data_root).resolve()
    directory = Path(manifest_dir) if manifest_dir else root / "training/manifests"

    # Verify manifest consistency between image workspace and DATA_DIR if both exist
    image_manifest_dir = (PROJECT_ROOT / "training/manifests").resolve()
    data_manifest_dir = (root / "training/manifests").resolve()
    if manifest_dir:
        image_manifest_dir = Path(manifest_dir).resolve()
    if image_manifest_dir.exists() and data_manifest_dir.exists() and image_manifest_dir != data_manifest_dir:
        for split_file in ("train.csv", "val.csv", "test.csv"):
            f_img = image_manifest_dir / split_file
            f_data = data_manifest_dir / split_file
            if f_img.exists() and f_data.exists():
                h_img = hashlib.sha256(f_img.read_bytes()).hexdigest()
                h_data = hashlib.sha256(f_data.read_bytes()).hexdigest()
                if h_img != h_data:
                    raise ValueError(
                        f"Manifest inconsistency between image copy ({f_img}: sha256={h_img[:12]}) "
                        f"and DATA_DIR copy ({f_data}: sha256={h_data[:12]}). Fail-closed."
                    )
    expected = approved_items(root)
    expected_keys = {tuple(str(item[k]) for k in FIELDS) for item in expected}
    if len(expected_keys) != 114:
        raise ValueError(f"Approved audit records do not identify 114 unique patches: {len(expected_keys)}")
    items = []
    counts = {}
    for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        with (directory / filename).open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        counts[split] = {"clean": 0, "defect": 0}
        for row in rows:
            if row["split"] != split or row["domain"] not in counts[split]:
                raise ValueError(f"Invalid split/domain in {filename}: {row}")
            if row["filename"] in ALL_EXCLUDED_DEFECT:
                raise ValueError(f"Excluded patch in manifest: {row['filename']}")
            full_path = resolve_patch_path(root, row["patch_path"])
            if not full_path.is_file():
                raise FileNotFoundError(full_path)
            items.append({**row, "full_path": full_path})
            counts[split][row["domain"]] += 1
        if counts[split] != EXPECTED_COUNTS[split]:
            raise ValueError(f"Approved split count mismatch: {split}: {counts[split]}")
    actual_keys = [tuple(str(item[k]) for k in FIELDS) for item in items]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        missing = expected_keys - set(actual_keys)
        extra = set(actual_keys) - expected_keys
        raise ValueError(f"Manifest differs from approved records. Missing={missing}, extra={extra}")
    validate_image_properties(items)
    check_data_leakage(items)
    replacements = sum(item.get("disposition") == "REPLACE_FROM_SOURCE_RGB" for item in expected)
    if replacements != 39:
        raise ValueError(f"Expected 39 approved clean replacements, got {replacements}")
    result = {"total": len(items), "counts": counts, "clean_replacements": replacements,
              "source_count": len({item["source_image"] for item in items}),
              "rgb_png_1024": True, "source_leakage": False, "excluded_patch_count": 0}
    print(json.dumps(result, indent=2))
    return result


def generate_manifests(data_root: Path, output_dir: Path) -> None:
    """Explicit generation into a new, separate directory; approved manifests stay immutable."""
    root, directory = Path(data_root).resolve(), Path(output_dir).resolve()
    protected = [root / name for name in ("training", "F971", "S942", "reports", "experiments")]
    if directory == root or any(directory.is_relative_to(p) for p in protected):
        raise ValueError("Manifest generation requires a separate output directory outside protected data")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Manifest output directory must be new or empty")
    items = approved_items(root)
    validate_image_properties(items)
    check_data_leakage(items)
    directory.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        with (directory / filename).open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: item[k] for k in FIELDS} for item in items if item["split"] == split)
    validate_baseline(root, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only validation of approved Defect-GAN RGB baseline")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_DIR", PROJECT_ROOT)))
    parser.add_argument("--manifest-dir", type=Path, help="Optional manifests in code image; images remain under DATA_DIR")
    parser.add_argument("--generate-manifests", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Required separate, empty output directory for explicit generation")
    args = parser.parse_args(argv)
    if args.generate_manifests:
        if args.output_dir is None:
            parser.error("--generate-manifests requires --output-dir")
        generate_manifests(args.data_root, args.output_dir)
    else:
        if args.output_dir is not None:
            parser.error("--output-dir applies only with --generate-manifests")
        validate_baseline(args.data_root, args.manifest_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
