"""Dataset statistics and visual preview reports for Defect-GAN crops."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def translate_segmentation(segmentation: list[list[float]], x: int, y: int) -> list[list[float]]:
    return [[coordinate - (x if index % 2 == 0 else y)
             for index, coordinate in enumerate(polygon)] for polygon in segmentation]


def write_bbox_statistics(coco: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    image_names = {image["id"]: image["file_name"] for image in coco["images"]}
    annotation_counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
    rows: list[dict[str, Any]] = []
    for annotation in sorted(coco["annotations"], key=lambda item: item["id"]):
        x, y, width, height = map(float, annotation["bbox"])
        rows.append({
            "annotation_id": annotation["id"],
            "image_id": annotation["image_id"],
            "filename": image_names[annotation["image_id"]],
            "bbox_x": round(x, 3),
            "bbox_y": round(y, 3),
            "bbox_width_px": round(width, 3),
            "bbox_height_px": round(height, 3),
            "scratch_length_px_aabb": round(max(width, height), 3),
            "scratch_width_px_aabb": round(min(width, height), 3),
            "aspect_ratio_aabb": round(max(width, height) / max(min(width, height), 1e-9), 4),
            "polygon_area_px2": round(float(annotation["area"]), 3),
            "annotation_count_on_source_image": annotation_counts[annotation["image_id"]],
        })
    write_csv(ROOT / "reports" / "scratch_bbox_statistics.csv", list(rows[0].keys()), rows)

    lengths = sorted(row["scratch_length_px_aabb"] for row in rows)
    widths = sorted(row["scratch_width_px_aabb"] for row in rows)
    areas = sorted(row["polygon_area_px2"] for row in rows)
    def percentile(values: list[float], proportion: float) -> float:
        return values[round((len(values) - 1) * proportion)]

    bins = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, float("inf"))]
    bin_lines = []
    for low, high in bins:
        label = f">={low}" if math.isinf(high) else f"{low}-{int(high) - 1}"
        count = sum(low <= value < high for value in lengths)
        bin_lines.append(f"- Chiều dài bbox {label} px: {count} annotation")
    summary = [
        "THỐNG KÊ KÍCH THƯỚC SCRATCH TRƯỚC KHI CROP",
        f"Số ảnh NG có annotation: {len(annotation_counts)}",
        f"Số annotation: {len(rows)}",
        "Chiều dài/chiều rộng được ước lượng bằng cạnh dài/cạnh ngắn của bbox axis-aligned; bbox gốc vẫn nằm trong CSV.",
        f"Scratch length bbox (px): min={min(lengths):.2f}, p25={percentile(lengths, .25):.2f}, median={percentile(lengths, .50):.2f}, p75={percentile(lengths, .75):.2f}, p90={percentile(lengths, .90):.2f}, p95={percentile(lengths, .95):.2f}, max={max(lengths):.2f}",
        f"Scratch width bbox (px): min={min(widths):.2f}, p25={percentile(widths, .25):.2f}, median={percentile(widths, .50):.2f}, p75={percentile(widths, .75):.2f}, p90={percentile(widths, .90):.2f}, p95={percentile(widths, .95):.2f}, max={max(widths):.2f}",
        f"Polygon area (px^2): min={min(areas):.2f}, median={percentile(areas, .50):.2f}, p90={percentile(areas, .90):.2f}, p95={percentile(areas, .95):.2f}, max={max(areas):.2f}",
        *bin_lines,
        f"BBox không thể vừa patch 512x512 theo kích thước gốc: {sum(row['bbox_width_px'] > 512 or row['bbox_height_px'] > 512 for row in rows)} annotations trước khi xét vùng lân cận.",
        f"BBox không thể vừa patch 1024x1024 theo kích thước gốc: {sum(row['bbox_width_px'] > 1024 or row['bbox_height_px'] > 1024 for row in rows)} annotations trước khi xét vùng lân cận.",
        f"Seed chia ảnh nguồn: {seed}; tỷ lệ train/validation/test: 70%/15%/15%.",
    ]
    (ROOT / "reports" / "scratch_bbox_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return rows


def write_split_manifest(split_map: dict[str, dict[str, str]], coco: dict[str, Any], seed: int) -> None:
    annotation_counts = Counter(annotation["image_id"] for annotation in coco["annotations"])
    ng_records = {image["file_name"]: image["id"] for image in coco["images"]}
    rows: list[dict[str, Any]] = []
    for source_type, mapping in split_map.items():
        for filename, split in sorted(mapping.items()):
            is_ng = source_type == "F971_NG"
            image_id = ng_records.get(filename, "") if is_ng else ""
            rows.append({
                "source_image": filename,
                "product_code": "F971",
                "source_status": "NG" if is_ng else "OK",
                "source_image_id": image_id,
                "split": split,
                "annotation_count": annotation_counts.get(image_id, 0) if image_id != "" else 0,
                "split_seed": seed,
            })
    write_csv(ROOT / "reports" / "source_image_splits.csv", list(rows[0].keys()), rows)


def make_previews(result: dict[str, Any], coco: dict[str, Any]) -> None:
    preview_dir = ROOT / "reports" / "patch_preview" / result["tag"]
    preview_dir.mkdir(parents=True, exist_ok=True)
    defects = [row for row in result["patches"] if row["label"] == "defect"]
    clean = [row for row in result["patches"] if row["label"] == "clean"]
    instance_data = json.loads((result["output_dir"] / "instances.json").read_text(encoding="utf-8"))
    annotations_by_patch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in instance_data["annotations"]:
        annotations_by_patch[annotation["image_id"]].append(annotation)
    source_annotations_by_id = {annotation["id"]: annotation for annotation in coco["annotations"]}
    patch_images_by_file = {image["file_name"]: image for image in instance_data["images"]}

    def patch_length(row: dict[str, Any]) -> float:
        image = next(item for item in instance_data["images"] if item["file_name"] == row["patch_file"])
        return max((max(annotation["bbox"][2], annotation["bbox"][3])
                    for annotation in annotations_by_patch[image["id"]]), default=0.0)

    size_order = sorted(defects, key=lambda row: (patch_length(row), row["source_image"], row["patch_file"]))
    if size_order:
        selected_defect_ids = {0, len(size_order) // 2, len(size_order) - 1}
        selected_defect_ids.add(max(range(len(size_order)), key=lambda index: size_order[index]["partial_neighbor_count"]))
        selected_defects = [size_order[index] for index in sorted(selected_defect_ids)]
    else:
        selected_defects = []
    selected_clean = clean[:3]
    selected = selected_defects + selected_clean
    if not selected:
        return

    cell_width, cell_height, caption_height = 384, 384, 48
    columns = 3
    rows_count = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows_count * (cell_height + caption_height)), "white")
    overlay_sheet = sheet.copy()

    for index, patch in enumerate(selected):
        x_cell = (index % columns) * cell_width
        y_cell = (index // columns) * (cell_height + caption_height)
        patch_path = result["output_dir"] / patch["patch_file"]
        with Image.open(patch_path) as image:
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x_cell, y_cell))
            overlay = image.copy()
        # Overlay annotations on the preview only; training patches remain untouched RGB PNGs.
        source_image = next((item for item in instance_data["images"] if item["file_name"] == patch["patch_file"]), None)
        if source_image is not None:
            scale_x = overlay.width / result["patch_size"]
            scale_y = overlay.height / result["patch_size"]
            drawer = ImageDraw.Draw(overlay)
            for annotation in annotations_by_patch[source_image["id"]]:
                for polygon in annotation["segmentation"]:
                    points = [(
                        float(polygon[i]) * scale_x,
                        float(polygon[i + 1]) * scale_y,
                    ) for i in range(0, len(polygon) - 1, 2)]
                    if len(points) >= 2:
                        drawer.line(points + [points[0]], fill=(255, 230, 0), width=2)
            for annotation_id in filter(None, patch["partial_neighbor_annotation_ids"].split("|")):
                source_annotation = source_annotations_by_id[int(annotation_id)]
                translated = translate_segmentation(
                    source_annotation["segmentation"], source_image["crop_x"], source_image["crop_y"]
                )
                for polygon in translated:
                    points = [(float(polygon[i]) * scale_x, float(polygon[i + 1]) * scale_y)
                              for i in range(0, len(polygon) - 1, 2)]
                    if len(points) >= 2:
                        drawer.line(points + [points[0]], fill=(255, 40, 40), width=2)
        overlay_sheet.paste(overlay, (x_cell, y_cell))
        if patch["label"] == "defect":
            first_line = f"DEFECT | {patch['split']} | full polygons={patch['annotation_count']} | edge={patch['partial_neighbor_count']}"
        else:
            first_line = f"CLEAN | {patch['split']}"
        for target in (sheet, overlay_sheet):
            draw = ImageDraw.Draw(target)
            draw.text((x_cell + 4, y_cell + cell_height + 3), first_line, fill="black", font=ImageFont.load_default())
            draw.text((x_cell + 4, y_cell + cell_height + 20), patch["source_image"][:56], fill="black", font=ImageFont.load_default())

    sheet.save(preview_dir / "rgb_samples.png", format="PNG", compress_level=3)
    overlay_sheet.save(preview_dir / "polygon_overlay_samples.png", format="PNG", compress_level=3)

    for index, patch in enumerate(selected_defects, start=1):
        source = result["output_dir"] / patch["patch_file"]
        shutil.copyfile(source, preview_dir / f"rgb_defect_{index:02d}.png")
        image_record = patch_images_by_file[patch["patch_file"]]
        with Image.open(source) as image:
            overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for annotation in annotations_by_patch[image_record["id"]]:
            for polygon in annotation["segmentation"]:
                points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=(255, 230, 0), width=max(2, result["patch_size"] // 256))
        for annotation_id in filter(None, patch["partial_neighbor_annotation_ids"].split("|")):
            source_annotation = source_annotations_by_id[int(annotation_id)]
            translated = translate_segmentation(
                source_annotation["segmentation"], image_record["crop_x"], image_record["crop_y"]
            )
            for polygon in translated:
                points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
                if len(points) >= 2:
                    draw.line(points + [points[0]], fill=(255, 40, 40), width=max(2, result["patch_size"] // 256))
        overlay.save(preview_dir / f"polygon_overlay_defect_{index:02d}.png", format="PNG", compress_level=3)
    for index, patch in enumerate(selected_clean, start=1):
        shutil.copyfile(result["output_dir"] / patch["patch_file"], preview_dir / f"rgb_clean_{index:02d}.png")


def write_patch_report(results: list[dict[str, Any]], split_map: dict[str, dict[str, str]], seed: int) -> None:
    lines = [
        "PATCH DATA PREPARATION REPORT",
        f"Root: {ROOT}",
        f"Seed: {seed}",
        "Split unit: source image; all patches from one source image belong to one split.",
        "Patch encoding: lossless RGB PNG, 8-bit, 3 channels; no resize or color conversion.",
        "Defect crops are created from F971 NG images; clean crops are sampled from F971 OK images.",
        "COCO polygon coordinates are translated into patch coordinates in each experiment's instances.json.",
        "No masks are saved. Spatial maps, if used in model code, must be generated internally from polygon coordinates.",
        "Each target polygon in a positive crop is complete. Other source polygons that intersect the crop but do not fit wholly are excluded from that crop's COCO and listed in reports/partial_neighbor_crops_512.csv or reports/partial_neighbor_crops_1024.csv.",
        "",
        "SOURCE IMAGE SPLITS",
    ]
    for split in SPLIT_RATIOS:
        ng_count = sum(value == split for value in split_map["F971_NG"].values())
        ok_count = sum(value == split for value in split_map["F971_OK"].values())
        lines.append(f"- {split}: F971 NG {ng_count} source images; F971 OK {ok_count} source images")
    for result in results:
        total = sum(row["total"] for row in result["split_counts"].values())
        positive = sum(row["positive"] for row in result["split_counts"].values())
        clean = sum(row["clean"] for row in result["split_counts"].values())
        lines.extend([
            "",
            f"EXPERIMENT {result['patch_size']}x{result['patch_size']}",
            f"- Output directory: {result['output_dir']}",
            f"- Total patches: {total}; defect: {positive}; clean: {clean}",
            f"- Defect prevalence: {positive / total:.1%}" if total else "- Defect prevalence: n/a",
            f"- Source annotations fully included: {result['covered_annotation_count']} / {result['covered_annotation_count'] + len(result['missing_annotation_ids'])}",
            f"- Annotations excluded without clipping: {len(result['exclusions'])}",
            f"- COCO patch annotations: {len(result['coco_annotations'])}",
            f"- Patches with a neighboring Scratch partially visible at an edge: {sum(row['partial_neighbor_count'] > 0 for row in result['patches'])}; see reports/partial_neighbor_crops_{result['patch_size']}.csv",
        ])
        for split, counts in result["split_counts"].items():
            lines.append(f"  - {split}: defect {counts['positive']}; clean {counts['clean']}; total {counts['total']}")
        if result["missing_annotation_ids"]:
            lines.append(f"- Excluded source annotation IDs: {len(result['missing_annotation_ids'])}; see reports/crop_exclusions_{result['patch_size']}.csv")
        lines.append(f"- Preview: reports/patch_preview/{result['tag']}/rgb_samples.png and polygon_overlay_samples.png")

    lines.extend([
        "",
        "PROPOSED TRAINING START POINT (not started)",
        "- 512x512 baseline: one NVIDIA GPU with 24 GB VRAM or more; start batch size 4 with mixed precision.",
        "- 1024x1024 comparison: same GPU; start batch size 1, use gradient accumulation for effective batch size 2-4.",
        "- Reduce batch size if peak memory exceeds the hardware; use gradient checkpointing before resizing patches.",
        "- Recommended model: RGB encoder-decoder Defect-GAN with SPADE spatial conditioning, learned noise injection,",
        "  defect-foreground plus alpha/spatial-map outputs, RGB compositing, PatchGAN source discriminator, and",
        "  the restoration/cycle path. Polygon-derived conditions remain tensors internal to training/inference.",
        "- Training has not been run.",
    ])
    (ROOT / "reports" / "patch_generation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
