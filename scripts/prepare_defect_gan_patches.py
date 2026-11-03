"""Prepare validated RGB crop experiments for the Defect-GAN dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from patch_reports import make_previews, write_bbox_statistics, write_patch_report, write_split_manifest


ROOT = Path(__file__).resolve().parents[1]
UNANNOTATED_IMAGE_ID = 34
UNANNOTATED_FILENAME = "2026-09-09-14-21-22.jpg"
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = 20261102


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    annotation_ids: tuple[int, ...]
    partial_annotation_ids: tuple[int, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def remove_unannotated_working_image(coco: dict[str, Any]) -> tuple[bool, str | None]:
    image_dir = ROOT / "F971" / "NG" / "images"
    coco_path = ROOT / "F971" / "NG" / "annotations" / "coco.json"
    manifest_path = ROOT / "reports" / "dataset_manifest.csv"
    audit_path = ROOT / "reports" / "annotation_audit.csv"
    image_path = image_dir / UNANNOTATED_FILENAME
    original_path = ROOT / "F971" / "NG" / "annotations" / "coco_original.json"
    original_hash = sha256(original_path)

    if any(annotation["image_id"] == UNANNOTATED_IMAGE_ID for annotation in coco["annotations"]):
        raise ValueError(f"Annotation unexpectedly references image_id={UNANNOTATED_IMAGE_ID}")

    matching = [image for image in coco["images"] if image["id"] == UNANNOTATED_IMAGE_ID]
    if matching and (len(matching) != 1 or matching[0]["file_name"] != UNANNOTATED_FILENAME):
        raise ValueError("Image ID 34 does not uniquely match the approved filename")

    if image_path.exists():
        recorded_sha = None
        if manifest_path.exists():
            _, manifest_rows = read_csv(manifest_path)
            rows = [row for row in manifest_rows if row["filename"] == UNANNOTATED_FILENAME]
            if len(rows) > 1:
                raise ValueError("Duplicate image rows found in dataset_manifest.csv")
            recorded_sha = rows[0]["sha256"] if rows else None
            if recorded_sha and sha256(image_path) != recorded_sha:
                raise ValueError("Working image checksum does not match its manifest row")

    if matching:
        coco["images"] = [image for image in coco["images"] if image["id"] != UNANNOTATED_IMAGE_ID]
        write_json(coco_path, coco)
    if image_path.exists():
        image_path.unlink()

    if manifest_path.exists():
        fields, rows = read_csv(manifest_path)
        write_csv(manifest_path, fields, [row for row in rows if row["filename"] != UNANNOTATED_FILENAME])
    if audit_path.exists():
        fields, rows = read_csv(audit_path)
        write_csv(
            audit_path,
            fields,
            [row for row in rows if row["filename"] != UNANNOTATED_FILENAME and row["image_id"] != str(UNANNOTATED_IMAGE_ID)],
        )

    if sha256(original_path) != original_hash:
        raise ValueError("coco_original.json changed during dataset cleanup")
    return bool(matching or image_path.exists()), original_hash


def validate_dataset(coco: dict[str, Any]) -> dict[str, Image.Image]:
    image_dir = ROOT / "F971" / "NG" / "images"
    records = {image["id"]: image for image in coco["images"]}
    if len(records) != len(coco["images"]):
        raise ValueError("COCO contains duplicate image IDs")
    if any(annotation["image_id"] not in records for annotation in coco["annotations"]):
        raise ValueError("COCO annotation points to a missing image record")
    if any(image["file_name"] == UNANNOTATED_FILENAME for image in coco["images"]):
        raise ValueError("Unannotated image record remains in normalized COCO")

    for annotation in coco["annotations"]:
        record = records[annotation["image_id"]]
        bbox = annotation.get("bbox", [])
        segmentation = annotation.get("segmentation", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"Invalid bbox in annotation {annotation['id']}")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > record["width"] or bbox[1] + bbox[3] > record["height"]:
            raise ValueError(f"BBox lies outside source image in annotation {annotation['id']}")
        if not segmentation or any(len(polygon) < 6 or len(polygon) % 2 for polygon in segmentation):
            raise ValueError(f"Invalid polygon in annotation {annotation['id']}")
        points = [
            (polygon[index], polygon[index + 1])
            for polygon in segmentation
            for index in range(0, len(polygon), 2)
        ]
        if any(x < 0 or y < 0 or x > record["width"] or y > record["height"] for x, y in points):
            raise ValueError(f"Polygon lies outside source image in annotation {annotation['id']}")
        tolerance = 1.0
        if (min(x for x, _ in points) < bbox[0] - tolerance
                or min(y for _, y in points) < bbox[1] - tolerance
                or max(x for x, _ in points) > bbox[0] + bbox[2] + tolerance
                or max(y for _, y in points) > bbox[1] + bbox[3] + tolerance):
            raise ValueError(f"BBox does not enclose polygon in annotation {annotation['id']}")

    opened: dict[str, Image.Image] = {}
    for record in coco["images"]:
        path = image_dir / record["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"COCO image is missing: {path}")
        with Image.open(path) as image:
            image.verify()
        image = Image.open(path)
        if image.mode != "RGB":
            image.close()
            raise ValueError(f"Expected RGB image, got {image.mode}: {path.name}")
        if image.size != (record["width"], record["height"]):
            image.close()
            raise ValueError(f"COCO dimensions mismatch for {path.name}")
        opened[record["file_name"]] = image

    for path in sorted((ROOT / "F971" / "OK" / "images").glob("*.jpg")):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB image, got {image.mode}: {path.name}")
    return opened


def annotation_bbox(annotation: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, width, height = annotation["bbox"]
    return float(x), float(y), float(x + width), float(y + height)


def safe_annotation_extent(
    annotation: dict[str, Any], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Add one pixel of margin for rounded COCO bbox values around polygon vertices."""
    x1, y1, x2, y2 = annotation_bbox(annotation)
    return max(0.0, x1 - 1), max(0.0, y1 - 1), min(float(image_width), x2 + 1), min(float(image_height), y2 + 1)


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def crop_origin(
    box: tuple[float, float, float, float], patch_size: int, image_width: int, image_height: int
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = box
    if x2 - x1 > patch_size or y2 - y1 > patch_size:
        return None
    if image_width < patch_size or image_height < patch_size:
        return None

    def axis_origin(start: float, end: float, image_length: int) -> int | None:
        lowest = max(0, math.ceil(end - patch_size))
        highest = min(math.floor(start), image_length - patch_size)
        if lowest > highest:
            return None
        ideal = round((start + end - patch_size) / 2)
        return min(max(ideal, lowest), highest)

    x = axis_origin(x1, x2, image_width)
    y = axis_origin(y1, y2, image_height)
    return None if x is None or y is None else (x, y)


def make_safe_crop(
    candidate: dict[str, Any], annotations: list[dict[str, Any]], patch_size: int,
    image_width: int, image_height: int,
) -> Crop | None:
    candidate_box = safe_annotation_extent(candidate, image_width, image_height)
    x_range = crop_origin(candidate_box, patch_size, image_width, image_height)
    if x_range is None:
        return None

    x_low = max(0, math.ceil(candidate_box[2] - patch_size))
    x_high = min(math.floor(candidate_box[0]), image_width - patch_size)
    y_low = max(0, math.ceil(candidate_box[3] - patch_size))
    y_high = min(math.floor(candidate_box[1]), image_height - patch_size)
    stride = max(16, patch_size // 16)
    ideal_x = x_range[0]
    ideal_y = crop_origin(candidate_box, patch_size, image_width, image_height)[1]

    def positions(low: int, high: int, ideal: int) -> list[int]:
        values = set(range(low, high + 1, stride))
        values.update((low, high, ideal))
        return sorted(values)

    best: tuple[int, int, float, int, int, int, int] | None = None
    best_ids: tuple[int, ...] = ()
    best_partial_ids: tuple[int, ...] = ()
    for y in positions(y_low, y_high, ideal_y):
        for x in positions(x_low, x_high, ideal_x):
            crop_box = (x, y, x + patch_size, y + patch_size)
            contained: list[int] = []
            partial: list[int] = []
            partial_visible_area = 0.0
            for annotation in annotations:
                box = safe_annotation_extent(annotation, image_width, image_height)
                if not bbox_intersects(box, crop_box):
                    continue
                is_contained = box[0] >= x and box[1] >= y and box[2] <= x + patch_size and box[3] <= y + patch_size
                if is_contained:
                    contained.append(annotation["id"])
                else:
                    partial.append(annotation["id"])
                    intersection_width = max(0.0, min(box[2], crop_box[2]) - max(box[0], crop_box[0]))
                    intersection_height = max(0.0, min(box[3], crop_box[3]) - max(box[1], crop_box[1]))
                    partial_visible_area += intersection_width * intersection_height
            if candidate["id"] not in contained:
                continue
            margin = min(
                candidate_box[0] - x,
                candidate_box[1] - y,
                x + patch_size - candidate_box[2],
                y + patch_size - candidate_box[3],
            )
            center_distance = abs(x - ideal_x) + abs(y - ideal_y)
            score = (len(contained), -len(partial), -partial_visible_area, round(margin), -center_distance, -y, -x)
            if best is None or score > best:
                best = score
                best_ids = tuple(sorted(contained))
                best_partial_ids = tuple(sorted(partial))
    return None if best is None else Crop(
        x=-best[6], y=-best[5], annotation_ids=best_ids, partial_annotation_ids=best_partial_ids
    )


def build_defect_crops(
    coco: dict[str, Any], image_records: dict[int, dict[str, Any]], patch_size: int
) -> tuple[dict[str, list[Crop]], list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    crops_by_file: dict[str, list[Crop]] = {}
    exclusions: list[dict[str, Any]] = []
    for image_id, annotations in sorted(by_image.items()):
        record = image_records[image_id]
        covered: set[int] = set()
        selected: list[Crop] = []
        ordered = sorted(
            annotations,
            key=lambda annotation: (annotation["bbox"][2] * annotation["bbox"][3], annotation["id"]),
            reverse=True,
        )
        for annotation in ordered:
            if annotation["id"] in covered:
                continue
            crop = make_safe_crop(annotation, annotations, patch_size, record["width"], record["height"])
            if crop is None:
                box = annotation_bbox(annotation)
                reason = "bbox_exceeds_patch" if (box[2] - box[0] > patch_size or box[3] - box[1] > patch_size) else "no_crop_window_found"
                exclusions.append({
                    "source_image_id": image_id,
                    "source_image": record["file_name"],
                    "annotation_id": annotation["id"],
                    "bbox_width_px": round(box[2] - box[0], 2),
                    "bbox_height_px": round(box[3] - box[1], 2),
                    "patch_size": patch_size,
                    "reason": reason,
                })
                continue
            selected.append(crop)
            covered.update(crop.annotation_ids)
        crops_by_file[record["file_name"]] = selected
        for annotation in annotations:
            if annotation["id"] not in covered and not any(row["annotation_id"] == annotation["id"] for row in exclusions):
                box = annotation_bbox(annotation)
                exclusions.append({
                    "source_image_id": image_id,
                    "source_image": record["file_name"],
                    "annotation_id": annotation["id"],
                    "bbox_width_px": round(box[2] - box[0], 2),
                    "bbox_height_px": round(box[3] - box[1], 2),
                    "patch_size": patch_size,
                    "reason": "not_covered_by_a_safe_crop",
                })
    covered_ids = {annotation_id for crops in crops_by_file.values() for crop in crops for annotation_id in crop.annotation_ids}
    exclusions = [row for row in exclusions if row["annotation_id"] not in covered_ids]
    return crops_by_file, exclusions


def split_source_images(
    coco: dict[str, Any], seed: int
) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {"F971_NG": {}, "F971_OK": {}}
    image_counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
    ng_records = sorted(coco["images"], key=lambda image: image["file_name"])
    ok_paths = sorted((ROOT / "F971" / "OK" / "images").glob("*.jpg"))

    def allocate(count: int) -> tuple[int, int, int]:
        values = [count * SPLIT_RATIOS[split] for split in ("train", "validation", "test")]
        result = [math.floor(value) for value in values]
        fractions = [value - math.floor(value) for value in values]
        for index in sorted(range(3), key=lambda item: fractions[item], reverse=True)[:count - sum(result)]:
            result[index] += 1
        if count >= 3:
            for index in (1, 2):
                if result[index] == 0:
                    donor = max(range(3), key=lambda item: result[item])
                    result[donor] -= 1
                    result[index] += 1
        return result[0], result[1], result[2]

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ng_records:
        count = image_counts[record["id"]]
        group = "one" if count == 1 else "two_to_five" if count <= 5 else "six_to_fifteen" if count <= 15 else "high_density"
        groups[group].append(record)
    for group_index, group_name in enumerate(("one", "two_to_five", "six_to_fifteen", "high_density")):
        values = list(groups[group_name])
        random.Random(seed + group_index).shuffle(values)
        if len(values) == 2:
            train_count, validation_count, test_count = 1, 0, 1
        else:
            train_count, validation_count, test_count = allocate(len(values))
        assignments = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        for record, split in zip(values, assignments, strict=True):
            mapping["F971_NG"][record["file_name"]] = split

    values = [{"file_name": path.name} for path in ok_paths]
    random.Random(seed + 1).shuffle(values)
    train_count, validation_count, test_count = allocate(len(values))
    assignments = ["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count
    for record, split in zip(values, assignments, strict=True):
        mapping["F971_OK"][record["file_name"]] = split

    return mapping


def clean_candidates(image_path: Path, patch_size: int) -> list[tuple[int, int]]:
    with Image.open(image_path) as image:
        width, height = image.size
    return [
        (x, y)
        for y in range(0, height - patch_size + 1, patch_size)
        for x in range(0, width - patch_size + 1, patch_size)
    ]


def translate_segmentation(segmentation: list[list[float]], x: int, y: int) -> list[list[float]]:
    return [[coordinate - (x if index % 2 == 0 else y)
             for index, coordinate in enumerate(polygon)] for polygon in segmentation]


def clear_generated_images(output_dir: Path) -> None:
    for split in SPLIT_RATIOS:
        for label in ("defect", "clean"):
            directory = output_dir / split / label
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob("*.png"):
                path.unlink()


def save_rgb_crop(source_path: Path, output_path: Path, x: int, y: int, size: int) -> None:
    with Image.open(source_path) as source:
        if source.mode != "RGB":
            raise ValueError(f"Source is not RGB: {source_path}")
        crop = source.crop((x, y, x + size, y + size))
        if crop.mode != "RGB" or crop.size != (size, size):
            raise ValueError(f"Invalid RGB crop generated from {source_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="PNG", compress_level=3)
        with Image.open(output_path) as saved:
            if saved.mode != "RGB" or saved.size != (size, size) or saved.tobytes() != crop.tobytes():
                raise ValueError(f"PNG crop did not preserve source pixels: {output_path}")


def generate_experiment(
    patch_size: int,
    coco: dict[str, Any],
    split_map: dict[str, dict[str, str]],
    seed: int,
) -> dict[str, Any]:
    tag = f"patches_{patch_size}x{patch_size}"
    output_dir = ROOT / "experiments" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_images(output_dir)

    records = {image["id"]: image for image in coco["images"]}
    records_by_name = {image["file_name"]: image for image in coco["images"]}
    defect_crops, exclusions = build_defect_crops(coco, records, patch_size)
    patches: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    truncated_rows: list[dict[str, Any]] = []
    next_patch_id = 1
    next_annotation_id = 1
    covered_annotations: set[int] = set()
    source_ng_splits = split_map["F971_NG"]

    for filename in sorted(defect_crops):
        record = records_by_name[filename]
        split = source_ng_splits[filename]
        source_path = ROOT / "F971" / "NG" / "images" / filename
        source_annotations = [annotation for annotation in coco["annotations"] if annotation["image_id"] == record["id"]]
        annotation_map = {annotation["id"]: annotation for annotation in source_annotations}
        for crop_number, crop in enumerate(defect_crops[filename], start=1):
            patch_name = f"{filename[:-4]}_crop{crop_number:03d}_x{crop.x}_y{crop.y}.png"
            relative_path = Path(split) / "defect" / patch_name
            patch_path = output_dir / relative_path
            save_rgb_crop(source_path, patch_path, crop.x, crop.y, patch_size)
            patch_id = next_patch_id
            next_patch_id += 1
            annotation_ids = list(crop.annotation_ids)
            for annotation_id in annotation_ids:
                annotation = annotation_map[annotation_id]
                bbox = list(annotation["bbox"])
                bbox[0] -= crop.x
                bbox[1] -= crop.y
                coco_annotations.append({
                    "id": next_annotation_id,
                    "image_id": patch_id,
                    "category_id": annotation["category_id"],
                    "bbox": bbox,
                    "iscrowd": annotation.get("iscrowd", 0),
                    "area": annotation["area"],
                    "segmentation": translate_segmentation(annotation["segmentation"], crop.x, crop.y),
                    "source_annotation_id": annotation_id,
                    "source_image_id": record["id"],
                })
                next_annotation_id += 1
                covered_annotations.add(annotation_id)
            coco_images.append({
                "id": patch_id,
                "license": record.get("license", 1),
                "file_name": relative_path.as_posix(),
                "height": patch_size,
                "width": patch_size,
                "source_image_id": record["id"],
                "source_image": filename,
                "split": split,
                "patch_kind": "defect",
                "crop_x": crop.x,
                "crop_y": crop.y,
            })
            patches.append({
                "patch_id": patch_id,
                "patch_file": relative_path.as_posix(),
                "split": split,
                "patch_size": patch_size,
                "label": "defect",
                "product_code": "F971",
                "source_status": "NG",
                "source_image_id": record["id"],
                "source_image": filename,
                "crop_x": crop.x,
                "crop_y": crop.y,
                "annotation_ids": "|".join(map(str, annotation_ids)),
                "annotation_count": len(annotation_ids),
                "partial_neighbor_annotation_ids": "|".join(map(str, crop.partial_annotation_ids)),
                "partial_neighbor_count": len(crop.partial_annotation_ids),
                "channels": 3,
                "color_mode": "RGB",
                "format": "PNG",
            })
            if crop.partial_annotation_ids:
                truncated_rows.append({
                    "patch_id": patch_id,
                    "patch_file": relative_path.as_posix(),
                    "split": split,
                    "source_image_id": record["id"],
                    "source_image": filename,
                    "fully_included_annotation_ids": "|".join(map(str, annotation_ids)),
                    "partial_neighbor_annotation_ids": "|".join(map(str, crop.partial_annotation_ids)),
                    "partial_neighbor_count": len(crop.partial_annotation_ids),
                })

    ok_paths = sorted((ROOT / "F971" / "OK" / "images").glob("*.jpg"))
    by_split_ok = {
        split: [path for path in ok_paths if split_map["F971_OK"][path.name] == split]
        for split in SPLIT_RATIOS
    }
    rng = random.Random(seed + patch_size)
    for split in SPLIT_RATIOS:
        target_negative = sum(row["split"] == split and row["label"] == "defect" for row in patches)
        candidates: list[tuple[Path, int, int]] = []
        for path in by_split_ok[split]:
            candidates.extend((path, x, y) for x, y in clean_candidates(path, patch_size))
        rng.shuffle(candidates)
        selected = candidates[:target_negative]
        if len(selected) < target_negative:
            raise ValueError(f"Not enough clean candidates in the {split} split")
        for source_path, x, y in selected:
            patch_name = f"{source_path.stem}_x{x}_y{y}.png"
            relative_path = Path(split) / "clean" / patch_name
            save_rgb_crop(source_path, output_dir / relative_path, x, y, patch_size)
            patch_id = next_patch_id
            next_patch_id += 1
            coco_images.append({
                "id": patch_id,
                "license": 1,
                "file_name": relative_path.as_posix(),
                "height": patch_size,
                "width": patch_size,
                "source_image_id": "",
                "source_image": source_path.name,
                "split": split,
                "patch_kind": "clean",
                "crop_x": x,
                "crop_y": y,
            })
            patches.append({
                "patch_id": patch_id,
                "patch_file": relative_path.as_posix(),
                "split": split,
                "patch_size": patch_size,
                "label": "clean",
                "product_code": "F971",
                "source_status": "OK",
                "source_image_id": "",
                "source_image": source_path.name,
                "crop_x": x,
                "crop_y": y,
                "annotation_ids": "",
                "annotation_count": 0,
                "partial_neighbor_annotation_ids": "",
                "partial_neighbor_count": 0,
                "channels": 3,
                "color_mode": "RGB",
                "format": "PNG",
            })

    if len(covered_annotations) != len(coco["annotations"]):
        missing = sorted({annotation["id"] for annotation in coco["annotations"]} - covered_annotations)
    else:
        missing = []
    write_csv(output_dir / "patch_manifest.csv", list(patches[0].keys()) if patches else [], patches)
    write_json(output_dir / "instances.json", {
        "info": {
            **coco.get("info", {}),
            "description": f"Lossless RGB {patch_size}x{patch_size} crop experiment; source-image split; polygon annotations preserved",
            "patch_size": patch_size,
            "split_seed": seed,
        },
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": coco_images,
        "annotations": coco_annotations,
    })
    validate_patch_coco(output_dir / "instances.json", output_dir, patch_size)
    write_csv(
        ROOT / "reports" / f"crop_exclusions_{patch_size}.csv",
        ["source_image_id", "source_image", "annotation_id", "bbox_width_px", "bbox_height_px", "patch_size", "reason"],
        exclusions,
    )
    write_csv(
        ROOT / "reports" / f"partial_neighbor_crops_{patch_size}.csv",
        ["patch_id", "patch_file", "split", "source_image_id", "source_image", "fully_included_annotation_ids", "partial_neighbor_annotation_ids", "partial_neighbor_count"],
        truncated_rows,
    )
    split_counts = {}
    for split in SPLIT_RATIOS:
        rows = [row for row in patches if row["split"] == split]
        split_counts[split] = {
            "positive": sum(row["label"] == "defect" for row in rows),
            "clean": sum(row["label"] == "clean" for row in rows),
            "total": len(rows),
        }
    return {
        "tag": tag,
        "patch_size": patch_size,
        "output_dir": output_dir,
        "patches": patches,
        "coco_images": coco_images,
        "coco_annotations": coco_annotations,
        "exclusions": exclusions,
        "covered_annotation_count": len(covered_annotations),
        "missing_annotation_ids": missing,
        "split_counts": split_counts,
    }


def validate_patch_coco(coco_path: Path, output_dir: Path, patch_size: int) -> None:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    image_ids = {image["id"] for image in data["images"]}
    annotation_ids = [annotation["id"] for annotation in data["annotations"]]
    if len(image_ids) != len(data["images"]) or len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError(f"Duplicate IDs found in {coco_path}")
    for image in data["images"]:
        if image["width"] != patch_size or image["height"] != patch_size:
            raise ValueError(f"Unexpected patch dimensions in {coco_path}: {image['file_name']}")
        if not (output_dir / image["file_name"]).is_file():
            raise FileNotFoundError(f"Patch COCO references missing image: {image['file_name']}")
    for annotation in data["annotations"]:
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"Patch annotation points to missing image: {annotation['id']}")
        x, y, width, height = annotation["bbox"]
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > patch_size or y + height > patch_size:
            raise ValueError(f"Translated patch bbox lies outside crop: {annotation['id']}")
        for polygon in annotation["segmentation"]:
            for index in range(0, len(polygon), 2):
                if not (0 <= polygon[index] <= patch_size and 0 <= polygon[index + 1] <= patch_size):
                    raise ValueError(f"Translated polygon lies outside crop: {annotation['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    coco_path = ROOT / "F971" / "NG" / "annotations" / "coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    original_path = ROOT / "F971" / "NG" / "annotations" / "coco_original.json"
    original_hash_before = sha256(original_path)
    remove_unannotated_working_image(coco)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    opened_ng = validate_dataset(coco)
    image_records = {image["id"]: image for image in coco["images"]}
    try:
        write_bbox_statistics(coco, args.seed)
        split_map = split_source_images(coco, args.seed)
        write_split_manifest(split_map, coco, args.seed)
        results = [
            generate_experiment(size, coco, split_map, args.seed)
            for size in (512, 1024)
        ]
        for result in results:
            make_previews(result, coco)
        write_patch_report(results, split_map, args.seed)
    finally:
        for image in opened_ng.values():
            image.close()

    if sha256(original_path) != original_hash_before:
        raise ValueError("coco_original.json changed during patch preparation")
    if (ROOT / "F971" / "NG" / "images" / UNANNOTATED_FILENAME).exists():
        raise ValueError("Approved unannotated working image still exists")
    final_coco = json.loads(coco_path.read_text(encoding="utf-8"))
    if len(final_coco["images"]) != 49 or len(final_coco["annotations"]) != 245:
        raise ValueError("Unexpected normalized COCO counts after cleanup")
    print("Prepared both RGB patch experiments successfully.")
    print(f"coco_original.json SHA-256 unchanged: {original_hash_before}")
    for result in results:
        print(f"{result['tag']}: {result['split_counts']}; exclusions={len(result['exclusions'])}")


if __name__ == "__main__":
    main()
