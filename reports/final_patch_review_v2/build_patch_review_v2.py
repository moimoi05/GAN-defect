from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "patches_1024x1024"
MANIFEST = EXP / "patch_manifest.csv"
COCO_PATCH = EXP / "instances.json"
COCO_SOURCE = ROOT / "F971" / "NG" / "annotations" / "coco.json"
BLACK_MAX = 5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Not RGB: {path} ({image.mode})")
        return np.asarray(image)


def ratios(array: np.ndarray) -> dict[str, float]:
    maxc = array.max(axis=2)
    lum = array.astype(np.float32).mean(axis=2)
    near_black = maxc < BLACK_MAX
    return {
        "near_black_pct": 100.0 * float(near_black.mean()),
        "nonblack_pct": 100.0 * float((~near_black).mean()),
        "dark_material_proxy_pct": 100.0 * float(((maxc >= BLACK_MAX) & (lum <= 120)).mean()),
        "midlight_proxy_pct": 100.0 * float((lum > 120).mean()),
        "mean_luminance": float(lum.mean()),
        "rgb_mean": [float(x) for x in array.reshape(-1, 3).mean(axis=0)],
    }


def image_rel_path(row: dict[str, str]) -> Path:
    return EXP / row["patch_file"]


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def segs_for_patch(patch_id: str, patch_data: dict) -> list[tuple[int, list[float]]]:
    image = next((x for x in patch_data["images"] if str(x.get("id")) == str(patch_id)), None)
    if not image:
        return []
    result = []
    for ann in patch_data["annotations"]:
        if str(ann.get("image_id")) == str(image["id"]):
            for seg in ann.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    result.append((int(ann.get("source_annotation_id", ann["id"])), seg))
    return result


def coco_polygons_by_source(coco: dict) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    images = {int(x["id"]): x for x in coco["images"]}
    anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns[int(ann["image_id"])].append(ann)
    return images, anns


def source_intersecting_polygons(
    image_id: int, x: int, y: int, size: int, coco_anns: dict[int, list[dict]]
) -> list[tuple[int, list[float], bool]]:
    found = []
    for ann in coco_anns.get(image_id, []):
        for seg in ann.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            pts = [(float(seg[i]) - x, float(seg[i + 1]) - y) for i in range(0, len(seg), 2)]
            if not any(0 <= px <= size and 0 <= py <= size for px, py in pts):
                bx, by, bw, bh = map(float, ann.get("bbox", [0, 0, 0, 0]))
                if bx + bw <= x or by + bh <= y or bx >= x + size or by >= y + size:
                    continue
            flat = [coord for pt in pts for coord in pt]
            fully_inside = all(0 <= px <= size and 0 <= py <= size for px, py in pts)
            found.append((int(ann["id"]), flat, fully_inside))
    return found


def draw_polygon_overlay(image: Image.Image, polygons: Iterable[tuple[int, list[float], bool]], line_width: int = 4) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for _, flat, full in polygons:
        pts = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=(255, 220, 0) if full else (255, 30, 30), width=line_width, joint="curve")
    return result


def tile_sheet(
    rows: list[dict[str, object]],
    out_path: Path,
    tile_size: int,
    columns: int,
    label_lines: list[tuple[str, ...]],
    image_key: str = "path",
    overlays: dict[str, list[tuple[int, list[float], bool]]] | None = None,
) -> None:
    label_h = 62
    margin = 8
    cell_w, cell_h = tile_size + 2 * margin, tile_size + label_h + 2 * margin
    nrows = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (columns * cell_w, nrows * cell_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    font = get_font(14)
    small = get_font(12)
    max_chars = max(34, int(tile_size / 7.5))
    for index, row in enumerate(rows):
        col, r = index % columns, index // columns
        left, top = col * cell_w + margin, r * cell_h + margin
        path = Path(str(row[image_key]))
        with Image.open(path) as im:
            if im.mode != "RGB":
                raise ValueError(f"Sheet input not RGB: {path}")
            shown = im.copy()
        key = str(row.get("patch_id", row.get("case_id", index)))
        if overlays and key in overlays:
            shown = draw_polygon_overlay(shown, overlays[key])
        if shown.size != (tile_size, tile_size):
            shown = shown.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        sheet.paste(shown, (left, top))
        raw_lines = label_lines[index]
        lines = []
        for line in raw_lines:
            lines.extend(textwrap.wrap(line, width=max_chars, break_long_words=True, break_on_hyphens=False) or [""])
        if len(lines) > 3:
            lines = lines[:2] + [lines[2][:max_chars - 1] + "…"]
        y0 = top + tile_size + 3
        for j, line in enumerate(lines[:3]):
            draw.text((left, y0 + j * 16), line, fill=(0, 0, 0), font=font if j == 0 else small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG", optimize=False, compress_level=3)


def best_clean_candidates(
    source_path: Path, size: int = 1024, stride: int = 128, count: int = 10,
    exclude_boxes: list[tuple[int, int, int]] | None = None,
) -> list[dict[str, object]]:
    arr = rgb_array(source_path)
    h, w = arr.shape[:2]
    if w < size or h < size:
        return []
    maxc = arr.max(axis=2)
    lum = arr.astype(np.float32).mean(axis=2)
    black = (maxc < BLACK_MAX).astype(np.uint8)
    dark = ((maxc >= BLACK_MAX) & (lum <= 120)).astype(np.uint8)
    def integral(mask: np.ndarray) -> np.ndarray:
        return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    bi, di = integral(black), integral(dark)
    area = size * size
    candidates = []
    ys = list(range(0, h - size + 1, stride))
    xs = list(range(0, w - size + 1, stride))
    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)
    for y in ys:
        for x in xs:
            nblack = bi[y + size, x + size] - bi[y, x + size] - bi[y + size, x] + bi[y, x]
            ndark = di[y + size, x + size] - di[y, x + size] - di[y + size, x] + di[y, x]
            black_pct = 100.0 * nblack / area
            dark_pct = 100.0 * ndark / area
            if black_pct <= 5 and dark_pct >= 35:
                # Prefer a mostly occupied dark coated surface and avoid a pure-background tile.
                score = black_pct + max(0.0, 55.0 - dark_pct) * 0.18
                candidates.append((score, black_pct, dark_pct, x, y))
    candidates.sort()
    selected = []
    exclude_boxes = exclude_boxes or []
    for score, black_pct, dark_pct, x, y in candidates:
        def overlaps_more_than_10(box: tuple[int, int, int]) -> bool:
            bx, by, bsize = box
            iw = max(0, min(x + size, bx + bsize) - max(x, bx))
            ih = max(0, min(y + size, by + bsize) - max(y, by))
            return (iw * ih) / (size * size) > 0.10
        if any(overlaps_more_than_10(box) for box in exclude_boxes):
            continue
        if all(abs(x - int(c["crop_x"])) >= size * 0.95 or abs(y - int(c["crop_y"])) >= size * 0.95 for c in selected):
            selected.append({"crop_x": x, "crop_y": y, "near_black_pct": black_pct, "dark_material_proxy_pct": dark_pct, "score": score})
        if len(selected) >= count:
            break
    return selected


def candidate_windows_for_annotations(
    arr: np.ndarray, polygons: list[list[float]], size: int = 1024, stride: int = 32
) -> list[dict[str, object]]:
    h, w = arr.shape[:2]
    xs_all = [poly[i] for poly in polygons for i in range(0, len(poly), 2)]
    ys_all = [poly[i + 1] for poly in polygons for i in range(0, len(poly), 2)]
    if not xs_all or max(xs_all) - min(xs_all) > size or max(ys_all) - min(ys_all) > size:
        return []
    minx, maxx, miny, maxy = min(xs_all), max(xs_all), min(ys_all), max(ys_all)
    low_x, high_x = max(0, math.ceil(maxx - size)), min(w - size, math.floor(minx))
    low_y, high_y = max(0, math.ceil(maxy - size)), min(h - size, math.floor(miny))
    if low_x > high_x or low_y > high_y:
        return []
    maxc = arr.max(axis=2)
    black = (maxc < BLACK_MAX).astype(np.int64)
    integral = np.pad(black, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h_candidates = []
    for y in range((low_y // stride) * stride, high_y + 1, stride):
        if y < low_y:
            continue
        for x in range((low_x // stride) * stride, high_x + 1, stride):
            if x < low_x:
                continue
            nblack = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            bp = 100.0 * nblack / (size * size)
            h_candidates.append({"crop_x": x, "crop_y": y, "near_black_pct": bp})
    # Add exact boundary choices that grid alignment can miss.
    for x, y in ((low_x, low_y), (high_x, high_y), (low_x, high_y), (high_x, low_y)):
        if not any(c["crop_x"] == x and c["crop_y"] == y for c in h_candidates):
            nblack = integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x]
            h_candidates.append({"crop_x": x, "crop_y": y, "near_black_pct": 100.0 * nblack / (size * size)})
    return sorted(h_candidates, key=lambda c: (c["near_black_pct"], c["crop_y"], c["crop_x"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_patches_rgb").mkdir(exist_ok=True)
    (OUT / "contact_sheets").mkdir(exist_ok=True)
    manifest = read_csv(MANIFEST)
    by_patch = {r["patch_id"]: r for r in manifest}
    patch_data = read_json(COCO_PATCH)
    coco = read_json(COCO_SOURCE)
    source_images, source_anns = coco_polygons_by_source(coco)
    source_by_name = {r["source_image"]: r for r in read_csv(ROOT / "reports" / "source_image_splits.csv")}
    clean_old = read_csv(ROOT / "reports" / "final_patch_review" / "clean_patch_visual_audit.csv")
    pending31 = read_csv(ROOT / "reports" / "final_patch_review" / "unreviewed_positive_1024.csv")
    edges = read_csv(ROOT / "reports" / "partial_neighbor_crops_1024.csv")
    oversize = read_csv(ROOT / "reports" / "crop_exclusions_1024.csv")

    # Audit every clean crop using RGB metrics and stage it for the required visual check.
    clean_rows = [r for r in manifest if r["label"] == "clean"]
    clean_audit = []
    clean_sheet_rows = []
    for row in clean_rows:
        path = image_rel_path(row)
        a = rgb_array(path)
        st = ratios(a)
        prior = next((x for x in clean_old if x["patch_id"] == row["patch_id"]), {})
        item = {
            "patch_id": row["patch_id"], "patch_name": Path(row["patch_file"]).name,
            "source_image": row["source_image"], "split": row["split"], "crop_x": row["crop_x"], "crop_y": row["crop_y"],
            "dimensions": "1024x1024", "mode": "RGB", "near_black_max_rgb_lt_5_pct": round(st["near_black_pct"], 3),
            "nonblack_proxy_pct": round(st["nonblack_pct"], 3), "dark_material_proxy_luma_le_120_pct": round(st["dark_material_proxy_pct"], 3),
            "mean_luminance": round(st["mean_luminance"], 2), "previous_visual_status": prior.get("visual_status", ""),
            "review_status": "REPLACE_REQUIRED" if st["near_black_pct"] >= 30 else "VISUAL_RECHECK_REQUIRED",
            "disposition": "REPLACE_OR_EXCLUDE" if st["near_black_pct"] >= 30 else "PENDING_VISUAL_QA",
            "note": "Near-black ratio is an objective empty-background proxy; final material suitability also requires visual inspection.",
        }
        clean_audit.append(item)
        clean_sheet_rows.append({"patch_id": row["patch_id"], "path": path, "name": Path(row["patch_file"]).name,
                                "source": row["source_image"], "split": row["split"], **st})
    clean_audit.sort(key=lambda r: float(r["near_black_max_rgb_lt_5_pct"]), reverse=True)
    write_csv(OUT / "clean_67_patch_pixel_audit.csv", clean_audit)
    clean_sheet_rows.sort(key=lambda r: float(r["near_black_pct"]), reverse=True)
    tile_sheet(clean_sheet_rows, OUT / "contact_sheets" / "clean_67_patch_audit.png", 384, 5,
               [(f"{x['name']} | black={x['near_black_pct']:.1f}%", f"src={x['source']} | {x['split']}") for x in clean_sheet_rows])

    # Any clean patch with >=30% near-black pixels is replaced by a fresh RGB crop when a
    # distinct product-surface window exists in the same source image. The specified four
    # user flags are included; this also catches other high-background rows found in the 67 audit.
    clean_candidate_rows = []
    clean_candidate_sheet = []
    audit_by_id = {str(x["patch_id"]): x for x in clean_audit}
    bad_clean = [x for x in clean_audit if float(x["near_black_max_rgb_lt_5_pct"]) >= 30]
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in bad_clean:
        by_source[str(row["source_image"])].append(row)
    manifest_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        manifest_by_source[row["source_image"]].append(row)
    for source_name, bad_rows in sorted(by_source.items()):
        source_path = ROOT / "F971" / "OK" / "images" / source_name
        bad_ids = {str(x["patch_id"]) for x in bad_rows}
        retained_boxes = []
        for old in manifest_by_source[source_name]:
            if old["patch_id"] in bad_ids:
                continue
            px = int(old["crop_x"])
            py = int(old["crop_y"])
            retained_boxes.append((px, py, 1024))
        candidates = best_clean_candidates(source_path, 1024, 128, max(20, len(bad_rows) * 8), retained_boxes)
        chosen: list[dict[str, object]] = []
        for bad in sorted(bad_rows, key=lambda x: (-float(x["near_black_max_rgb_lt_5_pct"]), int(x["patch_id"]))):
            candidate = next((c for c in candidates if all(
                abs(int(c["crop_x"]) - int(prev["crop_x"])) >= 1024 * 0.95 or
                abs(int(c["crop_y"]) - int(prev["crop_y"])) >= 1024 * 0.95 for prev in chosen)), None)
            if candidate is None:
                clean_candidate_rows.append({"flagged_patch_id": bad["patch_id"], "original_patch": bad["patch_name"],
                    "source_image": source_name, "split": bad["split"], "candidate_rank": "", "crop_x": "", "crop_y": "",
                    "near_black_pct": "", "dark_material_proxy_pct": "", "candidate_rgb_png": "",
                    "disposition": "EXCLUDE_NO_DISTINCT_LOW_BLACK_SURFACE_CROP"})
                continue
            chosen.append(candidate)
            out_name = f"clean_replacement_patch{bad['patch_id']}_x{candidate['crop_x']}_y{candidate['crop_y']}.png"
            out_path = OUT / "candidate_patches_rgb" / out_name
            with Image.open(source_path) as src:
                crop = src.crop((int(candidate["crop_x"]), int(candidate["crop_y"]),
                                 int(candidate["crop_x"]) + 1024, int(candidate["crop_y"]) + 1024))
                if crop.mode != "RGB" or crop.size != (1024, 1024):
                    raise ValueError(f"Bad clean candidate: {out_path}")
                crop.save(out_path, format="PNG", compress_level=1)
            metrics = ratios(rgb_array(out_path))
            clean_candidate_rows.append({"flagged_patch_id": bad["patch_id"], "original_patch": bad["patch_name"],
                "source_image": source_name, "split": bad["split"], "candidate_rank": 1,
                "crop_x": candidate["crop_x"], "crop_y": candidate["crop_y"],
                "near_black_pct": round(metrics["near_black_pct"], 3),
                "dark_material_proxy_pct": round(metrics["dark_material_proxy_pct"], 3),
                "candidate_rgb_png": out_name, "disposition": "VISUAL_REVIEW_PENDING"})
            clean_candidate_sheet.append({"patch_id": str(bad["patch_id"]), "path": out_path,
                "name": out_name, "source": source_name, "split": bad["split"],
                "near_black_pct": metrics["near_black_pct"], "crop_x": candidate["crop_x"], "crop_y": candidate["crop_y"]})
    write_csv(OUT / "clean_replacement_candidates.csv", clean_candidate_rows)
    tile_sheet(clean_candidate_sheet, OUT / "contact_sheets" / "clean_replacement_candidates.png", 384, 5,
               [(f"replace P{x['patch_id']} | black={x['near_black_pct']:.1f}% | x{x['crop_x']} y{x['crop_y']}",
                 f"src={x['source']} | {x['split']}", x["name"]) for x in clean_candidate_sheet])

    # Review every one of the 31 previously pending positive patches with full metadata and polygon outlines.
    positive_rows, positive_overlay, positive_labels = [], {}, []
    for row in pending31:
        p = by_patch[row["patch_id"]]
        path = image_rel_path(p)
        st = ratios(rgb_array(path))
        source_ids = row["annotation_ids"].replace("|", ",")
        positive_rows.append({"patch_id": row["patch_id"], "patch_name": row["patch_name"], "source_image": row["source_image"],
                              "annotation_ids": row["annotation_ids"], "split": row["split"], "path": path,
                              "near_black_pct": st["near_black_pct"]})
        positive_overlay[row["patch_id"]] = [(aid, seg, True) for aid, seg in segs_for_patch(row["patch_id"], patch_data)]
        positive_labels.append((f"P{row['patch_id']} | ann={source_ids} | black={st['near_black_pct']:.1f}%",
                                f"source={row['source_image']} | {row['split']}", row["patch_name"]))
    tile_sheet(positive_rows, OUT / "contact_sheets" / "scratch_31_pending_review.png", 512, 4,
               positive_labels, overlays=positive_overlay)

    # Edge crops: polygon is yellow when fully contained, red when cropped at the patch edge.
    edge_rows, edge_overlay, edge_labels = [], {}, []
    edge_proposals = []
    edge_dest = OUT / "candidate_patches_rgb" / "edge_replacement_1024_patch7_x2988_y352.png"
    edge7 = next(x for x in edges if x["patch_id"] == "7")
    edge7_manifest = by_patch["7"]
    edge_source = ROOT / "F971" / "NG" / "images" / edge7["source_image"]
    with Image.open(edge_source) as src:
        crop = src.crop((2988, 352, 4012, 1376))
        if crop.size != (1024, 1024) or crop.mode != "RGB":
            raise ValueError("Edge patch 7 replacement crop invalid")
        crop.save(edge_dest, format="PNG", compress_level=1)
    edge_proposals.append({"patch_id": "7", "source_image": edge7["source_image"], "split": edge7["split"], "crop_x": 2988, "crop_y": 352,
                           "candidate_rgb_png": edge_dest.name, "status": "REPLACE_CROP_GEOMETRICALLY_CONTAINS_ALL_NEIGHBORS; VISUAL_REVIEW_PENDING"})
    for row in edges:
        pm = by_patch[row["patch_id"]]
        path = image_rel_path(pm)
        img_id = int(row["source_image_id"])
        x, y = int(pm["crop_x"]), int(pm["crop_y"])
        polygons = source_intersecting_polygons(img_id, x, y, 1024, source_anns)
        edge_rows.append({"patch_id": row["patch_id"], "patch_name": Path(row["patch_file"]).name, "path": path,
                          "source_image": row["source_image"], "split": row["split"], "annotation_ids_fully_listed": row["fully_included_annotation_ids"],
                          "annotation_ids_partial": row["partial_neighbor_annotation_ids"], "near_black_pct": ratios(rgb_array(path))["near_black_pct"],
                          "polygon_count_intersecting": len(polygons), "fully_contained_polygon_count": sum(p[2] for p in polygons),
                          "clipped_polygon_count": sum(not p[2] for p in polygons),
                          "disposition": "REPLACE_WITH_1024_CANDIDATE" if row["patch_id"] == "7" else "DROP_FROM_STRICT_1024_BASELINE"})
        edge_overlay[row["patch_id"]] = polygons
        edge_labels.append((f"P{row['patch_id']} | yellow=full red=cut", f"src={row['source_image']} | {row['split']}",
                            f"partial IDs={row['partial_neighbor_annotation_ids'] or '-'}"))
    write_csv(OUT / "edge_patch_dispositions.csv", edge_rows)
    write_csv(OUT / "edge_replacement_candidates.csv", edge_proposals)
    tile_sheet(edge_rows, OUT / "contact_sheets" / "edge_16_full_yellow_clipped_red.png", 512, 4,
               edge_labels, image_key="path", overlays=edge_overlay)

    # NG candidate: preserve every vertex of annotation IDs 166-169 if a 1024 window permits.
    ng_row = next(x for x in manifest if Path(x["patch_file"]).name == "2026-09-10-16-10-51_crop001_x3482_y777.png")
    ng_src = ROOT / "F971" / "NG" / "images" / ng_row["source_image"]
    source_ann_map = {int(x["id"]): x for x in coco["annotations"]}
    ng_ann_ids = [int(x) for x in ng_row["annotation_ids"].split("|") if x]
    ng_polygons = [seg for aid in ng_ann_ids for seg in source_ann_map[aid].get("segmentation", []) if isinstance(seg, list)]
    with Image.open(ng_src) as src:
        ng_arr = np.asarray(src)
    ng_windows = candidate_windows_for_annotations(ng_arr, ng_polygons, 1024, 32)
    ng_candidate_rows = []
    ng_contact = []
    for rank, candidate in enumerate(ng_windows[:5], 1):
        name = f"ng_replacement_candidate{rank}_x{candidate['crop_x']}_y{candidate['crop_y']}.png"
        path = OUT / "candidate_patches_rgb" / name
        with Image.open(ng_src) as src:
            crop = src.crop((candidate["crop_x"], candidate["crop_y"], candidate["crop_x"] + 1024, candidate["crop_y"] + 1024))
            crop.save(path, format="PNG", compress_level=1)
        ng_candidate_rows.append({"original_patch": Path(ng_row["patch_file"]).name, "annotation_ids": ng_row["annotation_ids"],
                                  "source_image": ng_row["source_image"], "split": ng_row["split"], "candidate_rank": rank,
                                  "crop_x": candidate["crop_x"], "crop_y": candidate["crop_y"], "near_black_pct": round(float(candidate["near_black_pct"]), 3),
                                  "candidate_rgb_png": name, "disposition": "VISUAL_REVIEW_PENDING"})
        overlay = source_intersecting_polygons(int(ng_row["source_image_id"]), int(candidate["crop_x"]), int(candidate["crop_y"]), 1024, source_anns)
        ng_contact.append({"patch_id": str(rank), "path": path, "name": name, "source": ng_row["source_image"], "split": ng_row["split"],
                           "near_black_pct": candidate["near_black_pct"], "overlay": overlay, "ann_ids": ng_row["annotation_ids"]})
    write_csv(OUT / "ng_patch_replacement_candidates.csv", ng_candidate_rows)
    if ng_contact:
        tile_sheet(ng_contact, OUT / "contact_sheets" / "ng_patch_replacement_candidates.png", 512, 3,
                   [(f"Candidate {x['patch_id']} | ann={x['ann_ids']} | black={x['near_black_pct']:.1f}%", f"src={x['source']} | {x['split']}", x["name"]) for x in ng_contact],
                   overlays={str(x["patch_id"]): x["overlay"] for x in ng_contact})

    # Keep oversized annotations in COCO; emit source-native RGB proposals with polygon overlays in contact sheet only.
    oversized_rows, oversized_sheet = [], []
    proposals = {
        158: (1536, 661, 0), 159: (1536, 710, 102), 191: (1536, 732, 108), 202: (1536, 686, 0),
        204: (1536, 778, 155), 230: (1536, 2440, 0), 241: (1536, 620, 569), 245: (1536, 734, 196),
    }
    for row in oversize:
        aid = int(row["annotation_id"])
        size, x, y = proposals[aid]
        source_name = row["source_image"]
        image_meta = next((im for im in coco["images"] if int(im["id"]) == int(row["source_image_id"])), None)
        if image_meta is None:
            raise ValueError(f"COCO source image missing for annotation {aid}")
        source = ROOT / "F971" / "NG" / "images" / source_name
        out_name = f"oversize_ann{aid}_{size}_x{x}_y{y}.png"
        out_path = OUT / "candidate_patches_rgb" / out_name
        with Image.open(source) as src:
            crop = src.crop((x, y, x + size, y + size))
            if crop.mode != "RGB" or crop.size != (size, size):
                raise ValueError(f"Oversize crop invalid for {aid}: {crop.size} {crop.mode}")
            crop.save(out_path, format="PNG", compress_level=1)
        polygons = source_intersecting_polygons(int(row["source_image_id"]), x, y, size, source_anns)
        target = next((p for ann_id, p, full in polygons if ann_id == aid and full), None)
        if target is None:
            raise ValueError(f"Oversize annotation {aid} not fully included in proposal")
        st = ratios(rgb_array(out_path))
        split = source_by_name[source_name]["split"]
        oversized_rows.append({"annotation_id": aid, "source_image_id": row["source_image_id"], "source_image": source_name,
                               "split": split, "bbox_width_px": row["bbox_width_px"], "bbox_height_px": row["bbox_height_px"],
                               "proposal_size": size, "crop_x": x, "crop_y": y, "near_black_pct": round(st["near_black_pct"], 3),
                               "candidate_rgb_png": out_name, "annotation_retained_in_coco": "YES", "fully_contained": "YES",
                               "status": "SEPARATE_EXPERIMENT_NOT_1024_BASELINE"})
        oversized_sheet.append({"patch_id": str(aid), "path": out_path, "name": out_name, "source": source_name, "split": split,
                                "near_black_pct": st["near_black_pct"], "overlay": polygons, "ann_id": aid, "size": size})
    write_csv(OUT / "oversize_annotation_crop_proposals.csv", oversized_rows)
    tile_sheet(oversized_sheet, OUT / "contact_sheets" / "oversize_8_crop_proposals.png", 512, 4,
               [(f"ann={x['ann_id']} | {x['size']}² | black={x['near_black_pct']:.1f}%", f"src={x['source']} | {x['split']}", x["name"]) for x in oversized_sheet],
               overlays={x["patch_id"]: x["overlay"] for x in oversized_sheet})

    # Source-image leakage check for existing train/validation/test assignment.
    split_by_source: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        split_by_source[row["source_image"]].add(row["split"])
    leakage = {name: sorted(splits) for name, splits in split_by_source.items() if len(splits) > 1}
    leak_rows = [{"source_image": name, "splits": "|".join(splits), "leakage": "YES"} for name, splits in sorted(leakage.items())]
    write_csv(OUT / "source_split_leakage_check.csv", leak_rows or [{"source_image": "", "splits": "", "leakage": "NO"}])

    # Byte-level provenance: originals used here are read-only inputs; record their hashes for reproducibility.
    hashes = []
    for path in (MANIFEST, COCO_SOURCE, COCO_PATCH):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append({"path_relative_to_project": str(path.relative_to(ROOT)), "sha256": h})
    write_csv(OUT / "input_sha256_manifest.csv", hashes)

    # Compact machine-readable summary; manual KEEP/HOLD/REJECT dispositions are attached after visual review.
    with (OUT / "qa_build_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"clean_patches_audited": len(clean_audit), "pending_scratch_patches_contacted": len(positive_rows),
                   "edge_patches_audited": len(edge_rows), "oversize_annotations_retained": len(oversized_rows),
                   "source_level_split_leakage": bool(leakage), "training_started": False,
                   "all_saved_rgb_candidates_are_png": True, "near_black_definition": "RGB max channel < 5"}, stream, indent=2)


if __name__ == "__main__":
    main()
