"""Build native-resolution review sheets for the Defect-GAN patch sets."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "final_patch_review"
FONT_PATHS = [Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\segoeui.ttf")]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def spread_pick(rows: list[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row[key], row["source_image"], row["patch_file"]))
    if count >= len(ordered):
        return ordered
    if count <= 1:
        return [ordered[len(ordered) // 2]] if ordered else []
    indexes = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indexes]


def choose_distinct_sources(rows: list[dict[str, Any]], count: int, length_key: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    split_targets = {"train": round(count * 0.70), "validation": round(count * 0.15)}
    split_targets["test"] = count - split_targets["train"] - split_targets["validation"]
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        amount = split_targets[split]
        for item in spread_pick(split_rows, min(amount, len(split_rows)), length_key):
            if item["source_image"] not in used_sources:
                selected.append(item)
                used_sources.add(item["source_image"])
    remaining = [row for row in spread_pick(rows, len(rows), length_key) if row["source_image"] not in used_sources]
    for item in remaining:
        selected.append(item)
        used_sources.add(item["source_image"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected_ids = {row["patch_file"] for row in selected}
        selected.extend(row for row in spread_pick(rows, len(rows), length_key) if row["patch_file"] not in selected_ids)
    return selected[:count]


def paste_native_sheet(
    rows: list[dict[str, Any]],
    output_path: Path,
    patch_size: int,
    columns: int,
    caption_height: int,
    source_folder: Path,
    caption_fn,
    overlay_fn=None,
) -> None:
    cell_height = patch_size + caption_height
    row_count = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * patch_size, row_count * cell_height), (235, 235, 235))
    font = load_font(22 if patch_size >= 1024 else 17)
    for index, row in enumerate(rows):
        x = (index % columns) * patch_size
        y = (index // columns) * cell_height
        patch_path = source_folder / row["patch_file"]
        with Image.open(patch_path) as source:
            if source.mode != "RGB" or source.size != (patch_size, patch_size):
                raise ValueError(f"Unexpected patch mode/size: {patch_path}")
            tile = source.copy()
        if overlay_fn is not None:
            tile = overlay_fn(row, tile)
        sheet.paste(tile, (x, y))
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((x, y + patch_size, x + patch_size - 1, y + cell_height - 1), fill=(245, 245, 245))
        lines = caption_fn(row)
        for line_number, line in enumerate(lines[:3]):
            draw.text((x + 8, y + patch_size + 4 + line_number * 26), line, fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", compress_level=3)


def clip_segment(
    start: tuple[float, float], end: tuple[float, float], xmax: float, ymax: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Cohen-Sutherland clip of a polygon edge to a patch rectangle."""
    x1, y1 = start
    x2, y2 = end

    def code(x: float, y: float) -> int:
        result = 0
        if x < 0:
            result |= 1
        elif x > xmax:
            result |= 2
        if y < 0:
            result |= 4
        elif y > ymax:
            result |= 8
        return result

    code1, code2 = code(x1, y1), code(x2, y2)
    while True:
        if not (code1 | code2):
            return (x1, y1), (x2, y2)
        if code1 & code2:
            return None
        outcode = code1 or code2
        if outcode & 8:
            x, y = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1), ymax
        elif outcode & 4:
            x, y = x1 + (x2 - x1) * (0 - y1) / (y2 - y1), 0
        elif outcode & 2:
            x, y = xmax, y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
        else:
            x, y = 0, y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
        if outcode == code1:
            x1, y1, code1 = x, y, code(x, y)
        else:
            x2, y2, code2 = x, y, code(x, y)


def draw_polygon_outline(draw: ImageDraw.ImageDraw, polygon: list[float], color: tuple[int, int, int], width: int) -> None:
    points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
    if len(points) >= 2:
        draw.line(points + [points[0]], fill=color, width=width)


def make_edge_overlay(
    row: dict[str, Any], tile: Image.Image, image_by_file: dict[str, dict[str, Any]],
    anns_by_patch: dict[int, list[dict[str, Any]]], source_anns: dict[int, dict[str, Any]],
) -> Image.Image:
    image_record = image_by_file[row["patch_file"]]
    draw = ImageDraw.Draw(tile)
    line_width = 4
    for annotation in anns_by_patch[image_record["id"]]:
        for polygon in annotation["segmentation"]:
            draw_polygon_outline(draw, polygon, (255, 225, 0), line_width)
    partial_ids = [int(value) for value in row["partial_neighbor_annotation_ids"].split("|") if value]
    for annotation_id in partial_ids:
        annotation = source_anns[annotation_id]
        for polygon in annotation["segmentation"]:
            points = [(float(polygon[i]) - image_record["crop_x"], float(polygon[i + 1]) - image_record["crop_y"])
                      for i in range(0, len(polygon), 2)]
            for index in range(len(points)):
                segment = clip_segment(points[index], points[(index + 1) % len(points)], tile.width - 1, tile.height - 1)
                if segment is not None:
                    draw.line(segment, fill=(255, 35, 35), width=line_width)
    return tile


def make_roi_crop(annotation: dict[str, Any], image_record: dict[str, Any], source_path: Path, out_dir: Path) -> dict[str, Any]:
    x, y, width, height = annotation["bbox"]
    margin = int(max(64, max(width, height) * 0.25))
    with Image.open(source_path) as source:
        if source.mode != "RGB":
            raise ValueError(f"Non-RGB source image: {source_path}")
        left = max(0, int(math.floor(x)) - margin)
        top = max(0, int(math.floor(y)) - margin)
        right = min(source.width, int(math.ceil(x + width)) + margin)
        bottom = min(source.height, int(math.ceil(y + height)) + margin)
        raw = source.crop((left, top, right, bottom))
        if raw.mode != "RGB":
            raise ValueError(f"ROI crop changed RGB mode: {source_path}")
    stem = f"ann_{annotation['id']:03d}_{image_record['file_name'][:-4]}"
    raw_path = out_dir / f"{stem}_rgb.png"
    raw.save(raw_path, format="PNG", compress_level=3)
    overlay = raw.copy()
    draw = ImageDraw.Draw(overlay)
    for polygon in annotation["segmentation"]:
        shifted = [coordinate - (left if index % 2 == 0 else top)
                   for index, coordinate in enumerate(polygon)]
        draw_polygon_outline(draw, shifted, (255, 225, 0), 3)
    overlay_path = out_dir / f"{stem}_polygon_overlay.png"
    overlay.save(overlay_path, format="PNG", compress_level=3)
    return {
        "source_annotation_id": annotation["id"],
        "source_image": image_record["file_name"],
        "bbox_width_px": round(width, 2),
        "bbox_height_px": round(height, 2),
        "scratch_length_px": round(max(width, height), 2),
        "scratch_width_px": round(min(width, height), 2),
        "roi_x": left,
        "roi_y": top,
        "roi_width": raw.width,
        "roi_height": raw.height,
        "rgb_file": raw_path.name,
        "overlay_file": overlay_path.name,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp1024 = ROOT / "experiments" / "patches_1024x1024"
    exp512 = ROOT / "experiments" / "patches_512x512"
    coco_source = json.loads((ROOT / "F971" / "NG" / "annotations" / "coco.json").read_text(encoding="utf-8"))
    data1024 = json.loads((exp1024 / "instances.json").read_text(encoding="utf-8"))
    data512 = json.loads((exp512 / "instances.json").read_text(encoding="utf-8"))
    rows1024 = read_csv(exp1024 / "patch_manifest.csv")
    rows512 = read_csv(exp512 / "patch_manifest.csv")
    edges = read_csv(ROOT / "reports" / "partial_neighbor_crops_1024.csv")
    source_anns = {annotation["id"]: annotation for annotation in coco_source["annotations"]}
    coco_images1024 = {image["file_name"]: image for image in data1024["images"]}
    anns_by_patch1024: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data1024["annotations"]:
        anns_by_patch1024[annotation["image_id"]].append(annotation)

    annotation_length: dict[int, float] = {}
    for annotation in data1024["annotations"]:
        annotation_length[annotation["image_id"]] = max(
            annotation_length.get(annotation["image_id"], 0.0),
            float(max(annotation["bbox"][2], annotation["bbox"][3])),
        )
    for row in rows1024:
        row["partial_neighbor_count"] = int(row["partial_neighbor_count"])
        row["patch_id"] = int(row["patch_id"])
        row["patch_size"] = int(row["patch_size"])
        row["annotation_count"] = int(row["annotation_count"])
        row["crop_x"] = int(row["crop_x"])
        row["crop_y"] = int(row["crop_y"])
        row["max_scratch_length_px"] = annotation_length.get(row["patch_id"], 0.0)
    for row in rows512:
        row["partial_neighbor_count"] = int(row["partial_neighbor_count"])
        row["patch_id"] = int(row["patch_id"])
        row["annotation_count"] = int(row["annotation_count"])
        row["max_scratch_length_px"] = 0.0
    for annotation in data512["annotations"]:
        image_id = annotation["image_id"]
        row = next(item for item in rows512 if item["patch_id"] == image_id)
        row["max_scratch_length_px"] = max(row["max_scratch_length_px"], float(max(annotation["bbox"][2], annotation["bbox"][3])))

    positive1024 = [row for row in rows1024 if row["label"] == "defect"]
    clean1024 = [row for row in rows1024 if row["label"] == "clean"]
    edge_ids = {int(row["patch_id"]) for row in edges}
    clean_count_by_split = {split: sum(row["split"] == split for row in clean1024) for split in ("train", "validation", "test")}
    sample_clean: list[dict[str, Any]] = []
    sample_targets = {"train": 4, "validation": 3, "test": 3}
    used_clean_sources: set[str] = set()
    for split in ("train", "validation", "test"):
        split_rows = [row for row in clean1024 if row["split"] == split]
        selected = spread_pick(split_rows, min(sample_targets[split], len(split_rows)), "source_image")
        for row in selected:
            if row["source_image"] not in used_clean_sources:
                sample_clean.append(row)
                used_clean_sources.add(row["source_image"])
    for row in clean1024:
        if len(sample_clean) >= 10:
            break
        if row["source_image"] not in used_clean_sources:
            sample_clean.append(row)
            used_clean_sources.add(row["source_image"])

    eligible = [row for row in positive1024 if row["patch_id"] not in edge_ids]
    small = [row for row in eligible if row["max_scratch_length_px"] < 128]
    medium = [row for row in eligible if 128 <= row["max_scratch_length_px"] < 512]
    long = [row for row in eligible if row["max_scratch_length_px"] >= 512]
    picked = choose_distinct_sources(small, min(3, len(small)), "max_scratch_length_px")
    picked += choose_distinct_sources(medium, min(8, len(medium)), "max_scratch_length_px")
    picked += choose_distinct_sources(long, min(9, len(long)), "max_scratch_length_px")
    used_patch_files = {row["patch_file"] for row in picked}
    for row in choose_distinct_sources(eligible, len(eligible), "max_scratch_length_px"):
        if row["patch_file"] not in used_patch_files:
            picked.append(row)
            used_patch_files.add(row["patch_file"])
        if len(picked) >= 20:
            break
    sample_positive = picked[:20]
    if len(sample_positive) != 20 or len(sample_clean) != 10:
        raise ValueError("Could not select the requested 20 positive and 10 clean samples")

    sample_dir = OUT / "native_samples"
    for label in ("defect_1024", "clean_1024", "patch_512"):
        (sample_dir / label).mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []

    def collect(row: dict[str, Any], size_name: str) -> None:
        patch_dir = exp1024 if size_name == "1024" else exp512
        dest_dir = sample_dir / ("defect_1024" if row["label"] == "defect" and size_name == "1024" else "clean_1024" if size_name == "1024" else "patch_512")
        destination = dest_dir / Path(row["patch_file"]).name
        shutil.copyfile(patch_dir / row["patch_file"], destination)
        sample_rows.append({
            "experiment": f"{size_name}x{size_name}",
            "patch_id": row["patch_id"],
            "label": row["label"],
            "split": row["split"],
            "patch_name": Path(row["patch_file"]).name,
            "source_image": row["source_image"],
            "annotation_count": row["annotation_count"],
            "max_scratch_length_px": row["max_scratch_length_px"],
            "partial_neighbor_annotation_ids": row.get("partial_neighbor_annotation_ids", ""),
            "native_copy": destination.relative_to(OUT).as_posix(),
        })

    for row in sample_positive:
        collect(row, "1024")
    for row in sample_clean:
        collect(row, "1024")
    positives512 = [row for row in rows512 if row["label"] == "defect" and row["partial_neighbor_count"] == 0]
    clean512 = [row for row in rows512 if row["label"] == "clean"]
    sample512 = choose_distinct_sources(positives512, 5, "max_scratch_length_px")
    sample512 += choose_distinct_sources(clean512, 5, "max_scratch_length_px")
    for row in sample512:
        collect(row, "512")
    write_csv(OUT / "review_sample_manifest.csv", sample_rows)

    review1024 = sample_positive + sample_clean
    for row in review1024:
        row["partial_neighbor_annotation_ids"] = row.get("partial_neighbor_annotation_ids", "")
    paste_native_sheet(
        review1024,
        OUT / "contact_sheet_1024_20_scratch_10_clean.png",
        1024,
        5,
        86,
        exp1024,
        lambda row: (
            f"{row['label'].upper()} | patch: {Path(row['patch_file']).name}",
            f"source: {row['source_image']} | split: {row['split']} | annotations: {row['annotation_count']}",
        ),
    )

    def edge_overlay(row: dict[str, Any], tile: Image.Image) -> Image.Image:
        return make_edge_overlay(row, tile, coco_images1024, anns_by_patch1024, source_anns)

    edge_rows = []
    edge_patch_ids = {int(row["patch_id"]) for row in edges}
    selected_edges = [row for row in positive1024 if row["patch_id"] in edge_patch_ids]
    selected_edges.sort(key=lambda row: row["patch_id"])
    edge_metadata = {int(row["patch_id"]): row for row in edges}
    for row in selected_edges:
        info = edge_metadata[row["patch_id"]]
        row["partial_neighbor_annotation_ids"] = info["partial_neighbor_annotation_ids"]
        native = sample_dir / "edge_1024"
        native.mkdir(parents=True, exist_ok=True)
        source = exp1024 / row["patch_file"]
        shutil.copyfile(source, native / Path(row["patch_file"]).name)
        edge_rows.append({
            "patch_id": row["patch_id"],
            "patch_name": Path(row["patch_file"]).name,
            "source_image": row["source_image"],
            "split": row["split"],
            "fully_included_annotation_ids": info["fully_included_annotation_ids"],
            "partial_neighbor_annotation_ids": info["partial_neighbor_annotation_ids"],
            "partial_neighbor_count": info["partial_neighbor_count"],
            "recommended_action": "DROP_FROM_STRICT_BASELINE",
            "visual_review_status": "PENDING",
        })
    edge_caption = lambda row: (
        f"EDGE | {Path(row['patch_file']).name}",
        f"source: {row['source_image']} | full yellow; clipped red | split: {row['split']}",
        f"full polygons: {row['annotation_count']} | clipped neighbors: {row['partial_neighbor_count']}",
    )
    paste_native_sheet(
        selected_edges,
        OUT / "contact_sheet_1024_16_edge_scratch_overlay.png",
        1024,
        4,
        104,
        exp1024,
        edge_caption,
        edge_overlay,
    )
    for sheet_index in range(4):
        group = selected_edges[sheet_index * 4:(sheet_index + 1) * 4]
        paste_native_sheet(
            group,
            OUT / "edge_detail_sheets" / f"edge_detail_{sheet_index + 1:02d}.png",
            1024,
            2,
            104,
            exp1024,
            edge_caption,
            edge_overlay,
        )
    write_csv(OUT / "edge_case_review.csv", edge_rows)

    compare512_rows = sample512
    paste_native_sheet(
        compare512_rows,
        OUT / "contact_sheet_512_comparison_5_scratch_5_clean.png",
        512,
        5,
        72,
        exp512,
        lambda row: (
            f"{row['label'].upper()} | patch: {Path(row['patch_file']).name}",
            f"source: {row['source_image']} | split: {row['split']} | anns: {row['annotation_count']}",
        ),
    )

    clean_audit_rows = sorted(clean1024, key=lambda row: (row["split"], row["source_image"], row["patch_file"]))
    clean_review_manifest = []
    for index, row in enumerate(clean_audit_rows):
        clean_review_manifest.append({
            "index": index + 1,
            "patch_id": row["patch_id"],
            "patch_name": Path(row["patch_file"]).name,
            "source_image": row["source_image"],
            "split": row["split"],
            "coco_annotation_count": 0,
            "source_label": "F971 OK",
            "visual_status": "PENDING",
            "recommended_action": "PENDING_VISUAL_REVIEW",
            "notes": "",
        })
    write_csv(OUT / "clean_patch_visual_audit.csv", clean_review_manifest)
    clean_sheet_dir = OUT / "clean_audit_sheets"
    clean_sheet_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(clean_audit_rows), 9):
        group = clean_audit_rows[start:start + 9]
        paste_native_sheet(
            group,
            clean_sheet_dir / f"clean_audit_{start // 9 + 1:02d}.png",
            1024,
            3,
            78,
            exp1024,
            lambda row: (
                f"CLEAN | patch: {Path(row['patch_file']).name}",
                f"source: {row['source_image']} | split: {row['split']} | no COCO annotations",
            ),
        )

    annotation_by_id = {annotation["id"]: annotation for annotation in coco_source["annotations"]}
    image_by_id = {image["id"]: image for image in coco_source["images"]}
    native_crop_dir = OUT / "native_scratch_crops"
    native_crop_dir.mkdir(parents=True, exist_ok=True)
    lengths = {annotation["id"]: max(annotation["bbox"][2], annotation["bbox"][3])
               for annotation in coco_source["annotations"]}
    sorted_annotations = sorted(coco_source["annotations"], key=lambda annotation: lengths[annotation["id"]])
    buckets = [
        [annotation for annotation in sorted_annotations if lengths[annotation["id"]] < 64],
        [annotation for annotation in sorted_annotations if 64 <= lengths[annotation["id"]] < 128],
        [annotation for annotation in sorted_annotations if 128 <= lengths[annotation["id"]] < 256],
        [annotation for annotation in sorted_annotations if 256 <= lengths[annotation["id"]] < 512],
        [annotation for annotation in sorted_annotations if 512 <= lengths[annotation["id"]] < 1024],
        [annotation for annotation in sorted_annotations if lengths[annotation["id"]] >= 1024],
    ]
    native_annotations = [bucket[len(bucket) // 2] for bucket in buckets if bucket]
    native_roi_rows = []
    for annotation in native_annotations:
        image_record = image_by_id[annotation["image_id"]]
        source_path = ROOT / "F971" / "NG" / "images" / image_record["file_name"]
        native_roi_rows.append(make_roi_crop(annotation, image_record, source_path, native_crop_dir))
    write_csv(OUT / "native_scratch_crop_manifest.csv", native_roi_rows)

    for size, rows, experiment in ((1024, rows1024, exp1024), (512, rows512, exp512)):
        for row in rows:
            patch_path = experiment / row["patch_file"]
            with Image.open(patch_path) as image:
                if image.mode != "RGB" or image.size != (size, size):
                    raise ValueError(f"Patch is not native RGB {size}: {patch_path}")
    edge_sheet_count = len(selected_edges)
    summary = {
        "1024 source patches": len(rows1024),
        "1024 positive patches selected for representative sheet": len(sample_positive),
        "1024 clean patches selected for representative sheet": len(sample_clean),
        "1024 clean patches queued for full visual audit": len(clean_audit_rows),
        "1024 edge-partial patches": edge_sheet_count,
        "512 comparison patches": len(compare512_rows),
        "native Scratch ROI crops": len(native_roi_rows),
    }
    (OUT / "review_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
