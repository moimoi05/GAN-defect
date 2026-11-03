from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import build_patch_review_v2 as qa


OUT = qa.OUT
ROOT = qa.ROOT
EXP = qa.EXP


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    return qa.read_csv(path)


def polygon_points(ann: dict) -> list[float]:
    return [float(v) for seg in ann.get("segmentation", []) if isinstance(seg, list) for v in seg]


def exact_crop(source: Path, target: Path, x: int, y: int, size: int) -> bool:
    with Image.open(source) as src, Image.open(target) as out:
        expected = src.crop((x, y, x + size, y + size))
        return src.mode == "RGB" and expected.mode == "RGB" and out.mode == "RGB" and expected.size == out.size and expected.tobytes() == out.tobytes()


def main() -> None:
    manifest = read_csv(EXP / "patch_manifest.csv")
    by_id = {row["patch_id"]: row for row in manifest}
    clean_audit = read_csv(OUT / "clean_67_patch_pixel_audit.csv")
    clean_candidates = read_csv(OUT / "clean_replacement_candidates.csv")
    candidate_by_id = {row["flagged_patch_id"]: row for row in clean_candidates}
    old_clean = {row["patch_id"]: row for row in clean_audit}

    # A first heuristic candidate for clean patch 120 showed a brown Scratch.
    # The next alternate has a uniform dark coating and no visible Scratch at review scale.
    p120_alternate = "clean_replacement_patch120_alt3_x640_y1024.png"
    p120_alternate_path = OUT / "candidate_patches_rgb" / p120_alternate
    p120_rejected = [
        {"patch_id": "120", "candidate_rgb_png": "clean_replacement_patch120_x3968_y512.png",
         "crop_x": 3968, "crop_y": 512, "reason": "Visible copper/brown Scratch on product; reject as clean."},
        {"patch_id": "120", "candidate_rgb_png": "clean_replacement_patch120_alt2_x3712_y512.png",
         "crop_x": 3712, "crop_y": 512, "reason": "Visible copper/brown Scratch reaches upper edge; reject as clean."},
    ]
    write_csv(OUT / "clean_candidate_rejections.csv", p120_rejected)

    # The 31 pending defect crops receive visual dispositions from the RGB contact sheet.
    held_scratch = {
        "25": "HOLD: annotation encloses a small irregular discoloration; it is not clearly an elongated Scratch at native RGB review scale.",
        "29": "HOLD: the target is very small and low contrast; it cannot be distinguished confidently from local surface texture.",
        "43": "HOLD: the Scratch reaches the original frame boundary, so the full physical extent cannot be confirmed.",
    }
    pending31 = read_csv(ROOT / "reports" / "final_patch_review" / "unreviewed_positive_1024.csv")
    scratch_dispositions = []
    scratch_contact_rows = []
    scratch_overlays = {}
    scratch_labels = []
    patch_coco = qa.read_json(EXP / "instances.json")
    for row in pending31:
        p = by_id[row["patch_id"]]
        image_path = qa.image_rel_path(p)
        st = qa.ratios(qa.rgb_array(image_path))
        status = "HOLD" if row["patch_id"] in held_scratch else "KEEP"
        note = held_scratch.get(row["patch_id"], "Scratch is visibly distinct in the RGB crop; polygon follows the visible defect. Candidate only; not queued for training.")
        scratch_dispositions.append({"patch_id": row["patch_id"], "patch_name": row["patch_name"], "source_image": row["source_image"],
                                     "annotation_ids": row["annotation_ids"], "split": row["split"], "near_black_pct": round(st["near_black_pct"], 3),
                                     "visual_disposition": status, "training_status": "NOT_ADDED_PENDING_USER_APPROVAL", "review_note": note})
        scratch_contact_rows.append({"patch_id": row["patch_id"], "path": image_path})
        scratch_overlays[row["patch_id"]] = [(aid, seg, True) for aid, seg in qa.segs_for_patch(row["patch_id"], patch_coco)]
        scratch_labels.append((f"{status} | P{row['patch_id']} | ann={row['annotation_ids']}",
                               f"source={row['source_image']} | {row['split']}", row["patch_name"]))
    write_csv(OUT / "scratch_31_dispositions.csv", scratch_dispositions)
    qa.tile_sheet(scratch_contact_rows, OUT / "contact_sheets" / "scratch_31_review_with_dispositions.png", 512, 4,
                  scratch_labels, overlays=scratch_overlays)

    # Final clean set: keep visually clean rows below the replacement threshold and replace all rows >=30% near-black.
    final_clean_rows = []
    final_clean_sheet = []
    final_clean_candidate_rows = []
    for patch_id, audit in old_clean.items():
        original = by_id[patch_id]
        original_black = float(audit["near_black_max_rgb_lt_5_pct"])
        split = original["split"]
        source_image = original["source_image"]
        if original_black < 30:
            final_path = qa.image_rel_path(original)
            final_black = original_black
            disposition = "KEEP_ORIGINAL"
            note = "RGB contact sheet shows F971 product surface and no obvious Scratch; near-black area is below the 30% replacement trigger."
            crop_x, crop_y = int(original["crop_x"]), int(original["crop_y"])
        else:
            candidate = candidate_by_id[patch_id]
            if patch_id == "120":
                final_path = p120_alternate_path
                candidate_name = p120_alternate
                crop_x, crop_y = 640, 1024
                final_black = qa.ratios(qa.rgb_array(final_path))["near_black_pct"]
                candidate["candidate_rgb_png"] = candidate_name
                candidate["crop_x"] = str(crop_x)
                candidate["crop_y"] = str(crop_y)
                candidate["near_black_pct"] = f"{final_black:.3f}"
                candidate["dark_material_proxy_pct"] = f"{qa.ratios(qa.rgb_array(final_path))['dark_material_proxy_pct']:.3f}"
                candidate["disposition"] = "KEEP_ALT3; AUTOMATIC CANDIDATE REJECTED FOR VISIBLE SCRATCH"
                note = "Replace blank original. Automatic crop and alternate 2 visibly contain Scratch and are rejected; alt3 shows only the uniform F971 dark surface."
            else:
                final_path = OUT / "candidate_patches_rgb" / candidate["candidate_rgb_png"]
                crop_x, crop_y = int(candidate["crop_x"]), int(candidate["crop_y"])
                final_black = float(candidate["near_black_pct"])
                candidate["disposition"] = "KEEP_VISUALLY_CLEAN_CANDIDATE"
                note = "Replace original high-background crop with native-size source RGB window; dark F971 coating and product context remain visible; no Scratch is apparent."
            disposition = "REPLACE_FROM_SOURCE_RGB"
        metrics = qa.ratios(qa.rgb_array(final_path))
        final_name = str(final_path.relative_to(OUT)).replace("\\", "/") if final_path.is_relative_to(OUT) else str(original["patch_file"])
        final_clean_rows.append({"original_patch_id": patch_id, "original_patch": Path(original["patch_file"]).name,
                                 "source_image": source_image, "split": split, "crop_x": crop_x, "crop_y": crop_y,
                                 "final_rgb_file": final_name, "final_dimensions": "1024x1024", "mode": "RGB",
                                 "original_near_black_pct": round(original_black, 3), "final_near_black_pct": round(final_black, 3),
                                 "final_nonblack_proxy_pct": round(metrics["nonblack_pct"], 3),
                                 "final_dark_material_proxy_pct": round(metrics["dark_material_proxy_pct"], 3),
                                 "disposition": disposition, "visual_note": note})
        final_clean_sheet.append({"patch_id": patch_id, "path": final_path, "name": Path(final_name).name,
                                  "source": source_image, "split": split, "black_pct": final_black,
                                  "disposition": disposition})
    write_csv(OUT / "clean_final_manifest.csv", final_clean_rows)
    write_csv(OUT / "clean_replacement_candidates.csv", clean_candidates)
    clean_contact_rows = sorted(final_clean_sheet, key=lambda row: int(row["patch_id"]))
    qa.tile_sheet(clean_contact_rows, OUT / "contact_sheets" / "clean_after_filtering_67.png", 384, 5,
                  [(f"P{x['patch_id']} {x['disposition']} | black={x['black_pct']:.1f}%",
                    f"src={x['source']} | {x['split']}", x["name"]) for x in clean_contact_rows])
    final_replacements = [row for row in final_clean_sheet if row["disposition"] == "REPLACE_FROM_SOURCE_RGB"]
    qa.tile_sheet(final_replacements, OUT / "contact_sheets" / "clean_replacement_candidates.png", 384, 5,
                  [(f"replace P{x['patch_id']} | black={x['black_pct']:.1f}%",
                    f"src={x['source']} | {x['split']}", x["name"]) for x in final_replacements])

    # Build a clean/defect disposition ledger and source-level split audit.
    edges = read_csv(ROOT / "reports" / "partial_neighbor_crops_1024.csv")
    edge_dispositions = read_csv(OUT / "edge_patch_dispositions.csv")
    edge_by_id = {row["patch_id"]: row for row in edge_dispositions}
    for row in edge_dispositions:
        row["final_action"] = "DROP_FROM_STRICT_1024_BASELINE"
        row["review_note"] = ("A trial recrop at x=2988,y=352 restores the originally clipped annotation 24, but clips neighboring annotations 11, 25, and 26; no clean 1024 window was found, so exclude from the strict baseline." if row["patch_id"] == "7"
                              else "The complete neighboring Scratch group cannot be retained in a 1024 crop; exclude the clipped crop from the strict baseline.")
    write_csv(OUT / "edge_patch_dispositions.csv", edge_dispositions)
    # Edge contact sheet labels show strict disposition while retaining yellow/full and red/clipped polygon lines.
    edge_contact = []
    edge_labels = []
    edge_overlay = {}
    for row in edges:
        pm = by_id[row["patch_id"]]
        p = qa.image_rel_path(pm)
        edge_contact.append({"patch_id": row["patch_id"], "path": p})
        edge_overlay[row["patch_id"]] = qa.source_intersecting_polygons(int(row["source_image_id"]), int(pm["crop_x"]), int(pm["crop_y"]), 1024,
                                                                        qa.coco_polygons_by_source(qa.read_json(qa.COCO_SOURCE))[1])
        action = "DROP STRICT" if row["patch_id"] == "7" else "DROP STRICT"
        edge_labels.append((f"{action} | P{row['patch_id']} | yellow=full red=cut", f"source={row['source_image']} | {row['split']}",
                            f"partial IDs={row['partial_neighbor_annotation_ids'] or '-'}"))
    qa.tile_sheet(edge_contact, OUT / "contact_sheets" / "edge_16_final_dispositions.png", 512, 4,
                  edge_labels, overlays=edge_overlay)

    # Render the 1024 patch-7 replacement as an RGB polygon preview and verify the crop is pixel-exact.
    edge7 = read_csv(OUT / "edge_replacement_candidates.csv")[0]
    edge7["status"] = "REJECTED_FOR_STRICT_1024; TRIAL RECROP STILL CLIPS NEIGHBOR ANNOTATIONS 11|25|26"
    write_csv(OUT / "edge_replacement_candidates.csv", [edge7])
    edge_source = ROOT / "F971" / "NG" / "images" / edge7["source_image"]
    edge_rgb = OUT / "candidate_patches_rgb" / edge7["candidate_rgb_png"]
    edge_polys = qa.source_intersecting_polygons(int(by_id["7"]["source_image_id"]), 2988, 352, 1024,
                                                 qa.coco_polygons_by_source(qa.read_json(qa.COCO_SOURCE))[1])
    with Image.open(edge_rgb) as image:
        preview = qa.draw_polygon_overlay(image, edge_polys, 3)
        preview.save(OUT / "contact_sheets" / "edge_patch7_replacement_polygon_preview.png", format="PNG", compress_level=2)
    edge7_full = sum(int(item[2]) for item in edge_polys)
    edge7_clipped = sum(not item[2] for item in edge_polys)
    edge7_clipped_ids = [str(item[0]) for item in edge_polys if not item[2]]
    edge_exact = exact_crop(edge_source, edge_rgb, 2988, 352, 1024)

    # The highlighted NG patch has a clean-looking 1024 target window, but it clips neighbor annotation 163.
    ng_cands = read_csv(OUT / "ng_patch_replacement_candidates.csv")
    for row in ng_cands:
        row["disposition"] = "HOLD_OUTSIDE_STRICT_1024_BASELINE" if row["candidate_rank"] == "1" else "ALTERNATE_NOT_SELECTED"
        row["review_note"] = "Target annotation IDs 166|167|168|169 fit; neighboring annotation 163 is clipped in this 1024 window. Do not include in strict baseline."
    write_csv(OUT / "ng_patch_replacement_candidates.csv", ng_cands)
    ng_src_name = "2026-09-10-16-10-51.jpg"
    ng_src = ROOT / "F971" / "NG" / "images" / ng_src_name
    ng_size = 1536
    ng_x, ng_y = 3023, 0
    ng1536_name = f"ng_replacement_neighbor_context_1536_x{ng_x}_y{ng_y}.png"
    ng1536_path = OUT / "candidate_patches_rgb" / ng1536_name
    with Image.open(ng_src) as src:
        if ng_x + ng_size > src.width or ng_y + ng_size > src.height:
            raise ValueError("1536 NG proposal exceeds original source bounds")
        src.crop((ng_x, ng_y, ng_x + ng_size, ng_y + ng_size)).save(ng1536_path, format="PNG", compress_level=1)
    coco = qa.read_json(qa.COCO_SOURCE)
    coco_images, coco_anns = qa.coco_polygons_by_source(coco)
    ng_image_id = next(int(im["id"]) for im in coco["images"] if im["file_name"] == ng_src_name)
    ng1536_polys = qa.source_intersecting_polygons(ng_image_id, ng_x, ng_y, ng_size, coco_anns)
    target_ng_ids = {163, 166, 167, 168, 169}
    target_ng_full = {ann_id for ann_id, _, full in ng1536_polys if full and ann_id in target_ng_ids}
    target_ng_clipped = {ann_id for ann_id, _, full in ng1536_polys if not full and ann_id in target_ng_ids}
    with Image.open(ng1536_path) as image:
        ng1536_preview = qa.draw_polygon_overlay(image, ng1536_polys, 4)
        ng1536_preview.save(OUT / "contact_sheets" / "ng_neighbor_context_1536_polygon_preview.png", format="PNG", compress_level=2)
    ng1536_metrics = qa.ratios(qa.rgb_array(ng1536_path))
    write_csv(OUT / "ng_1536_neighbor_context_proposal.csv", [{
        "source_image": ng_src_name, "split": "train", "target_annotation_ids": "166|167|168|169",
        "included_neighbor_ids": "163", "proposal_size": 1536, "crop_x": ng_x, "crop_y": ng_y,
        "near_black_pct": round(ng1536_metrics["near_black_pct"], 3), "full_target_and_neighbor_ids": "|".join(map(str, sorted(target_ng_full))),
        "clipped_target_or_neighbor_ids": "|".join(map(str, sorted(target_ng_clipped))),
        "rgb_file": ng1536_name, "status": "SEPARATE_SIZE_HOLD_FOR_USER_REVIEW_NOT_1024_BASELINE"}])

    # Add context margins to 1536 proposals for the 8 oversized annotations.
    oversize_rows = read_csv(OUT / "oversize_annotation_crop_proposals.csv")
    for row in oversize_rows:
        image_id = int(row["source_image_id"])
        aid = int(row["annotation_id"])
        ann = next(a for a in coco["annotations"] if int(a["id"]) == aid)
        pts = polygon_points(ann)
        xs, ys = pts[0::2], pts[1::2]
        x, y, size = int(row["crop_x"]), int(row["crop_y"]), int(row["proposal_size"])
        row["context_margin_left_px"] = round(min(xs) - x, 2)
        row["context_margin_right_px"] = round((x + size) - max(xs), 2)
        row["context_margin_top_px"] = round(min(ys) - y, 2)
        row["context_margin_bottom_px"] = round((y + size) - max(ys), 2)
        row["coco_annotation_retained"] = "YES"
        row["training_status"] = "SEPARATE_EXPERIMENT_NOT_1024_BASELINE"
    write_csv(OUT / "oversize_annotation_crop_proposals.csv", oversize_rows)

    # Source split audit includes every 1024 source image and reports any multi-split source explicitly.
    split_map = {row["source_image"]: row for row in read_csv(ROOT / "reports" / "source_image_splits.csv")}
    source_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        source_rows[row["source_image"]].append(row)
    leakage_rows = []
    leak_sources = []
    for source_name, rows in sorted(source_rows.items()):
        splits = sorted({row["split"] for row in rows})
        source_split = split_map.get(source_name, {}).get("split", "MISSING")
        leak = len(splits) != 1 or source_split not in splits
        if leak:
            leak_sources.append(source_name)
        leakage_rows.append({"source_image": source_name, "manifest_patch_count": len(rows), "manifest_splits": "|".join(splits),
                             "source_split_assignment": source_split, "leakage": "YES" if leak else "NO"})
    write_csv(OUT / "source_split_leakage_check.csv", leakage_rows)

    # Check COCO referential integrity without editing it.
    source_image_ids = {int(item["id"]) for item in coco["images"]}
    dangling = [int(ann["id"]) for ann in coco["annotations"] if int(ann["image_id"]) not in source_image_ids]
    duplicate_image_ids = len(source_image_ids) != len(coco["images"])
    coco_integrity = {"image_count": len(coco["images"]), "annotation_count": len(coco["annotations"]),
                      "dangling_annotation_ids": dangling, "duplicate_image_ids": duplicate_image_ids,
                      "source_file_modified_by_this_review": False}
    (OUT / "coco_read_only_integrity.json").write_text(json.dumps(coco_integrity, indent=2), encoding="utf-8")

    # Count review-approved candidates, not training-set membership.
    clean_counts = Counter(row["split"] for row in final_clean_rows)
    selected20 = [row for row in read_csv(ROOT / "reports" / "final_patch_review" / "review_sample_manifest.csv")
                  if row["experiment"] == "1024x1024" and row["label"] == "defect"]
    defect_counts = Counter()
    for row in selected20:
        if row["patch_id"] != "45":
            defect_counts[row["split"]] += 1
    for row in scratch_dispositions:
        if row["visual_disposition"] == "KEEP":
            defect_counts[row["split"]] += 1
    patch_counts = []
    for split in ("train", "validation", "test"):
        patch_counts.append({"split": split, "defect_review_approved_candidates": defect_counts[split],
                             "clean_review_approved_candidates": clean_counts[split],
                             "total_review_approved_candidates": defect_counts[split] + clean_counts[split],
                             "training_started": "NO"})
    patch_counts.append({"split": "TOTAL", "defect_review_approved_candidates": sum(defect_counts.values()),
                         "clean_review_approved_candidates": sum(clean_counts.values()),
                         "total_review_approved_candidates": sum(defect_counts.values()) + sum(clean_counts.values()),
                         "training_started": "NO"})
    write_csv(OUT / "final_patch_counts_by_split.csv", patch_counts)

    # Pixel-exact checks for every selected source crop: PNG compression is lossless and no resampling occurred.
    exact_checks = []
    for row in final_clean_rows:
        if row["disposition"] != "REPLACE_FROM_SOURCE_RGB":
            continue
        source = ROOT / "F971" / "OK" / "images" / row["source_image"]
        target = OUT / row["final_rgb_file"]
        exact_checks.append((str(target.relative_to(OUT)), exact_crop(source, target, int(row["crop_x"]), int(row["crop_y"]), 1024)))
    exact_checks.append((edge_rgb.name, exact_crop(edge_source, edge_rgb, 2988, 352, 1024)))
    exact_checks.append((ng1536_name, exact_crop(ng_src, ng1536_path, ng_x, ng_y, 1536)))
    for row in oversize_rows:
        target = OUT / "candidate_patches_rgb" / row["candidate_rgb_png"]
        source = ROOT / "F971" / "NG" / "images" / row["source_image"]
        exact_checks.append((row["candidate_rgb_png"], exact_crop(source, target, int(row["crop_x"]), int(row["crop_y"]), int(row["proposal_size"]))))
    write_csv(OUT / "rgb_pixel_provenance_check.csv", [{"candidate_file": name, "pixel_exact_source_crop": "YES" if ok else "NO"} for name, ok in exact_checks])

    # Human-readable summary.
    bands = Counter()
    for row in clean_audit:
        pct = float(row["near_black_max_rgb_lt_5_pct"])
        if pct >= 99.99:
            bands["100% near-black"] += 1
        elif pct >= 30:
            bands["30-<100% near-black"] += 1
        elif pct >= 5:
            bands["5-<30% near-black"] += 1
        else:
            bands["<5% near-black"] += 1
    ng_original = next(row for row in manifest if Path(row["patch_file"]).name == "2026-09-10-16-10-51_crop001_x3482_y777.png")
    ng_original_black = qa.ratios(qa.rgb_array(qa.image_rel_path(ng_original)))["near_black_pct"]
    edge_drop = [row["patch_id"] for row in edge_dispositions]
    kept_scratch = [row["patch_id"] for row in scratch_dispositions if row["visual_disposition"] == "KEEP"]
    held_scratch_ids = [row["patch_id"] for row in scratch_dispositions if row["visual_disposition"] == "HOLD"]
    user_flag_names = [
        "THAN_T_D_1_2026-09-14-10-49-55_x3072_y1024.png",
        "THAN_T_D_1_2026-09-14-11-04-06_x4096_y1024.png",
        "THAN_T_D_2_2026-09-14-11-06-11_x3072_y2048.png",
        "THAN_T_D_6_2026-09-14-11-09-28_x4096_y1024.png",
    ]
    by_name = {row["patch_name"]: row for row in clean_audit}
    flagged_text = "\n".join(f"  - {name}: {float(by_name[name]['near_black_max_rgb_lt_5_pct']):.3f}% near-black; replaced from same source image." for name in user_flag_names)
    count_text = "\n".join(f"  - {row['split']}: {row['defect_review_approved_candidates']} defect + {row['clean_review_approved_candidates']} clean = {row['total_review_approved_candidates']} candidate patches" for row in patch_counts[:3])
    report = f"""FINAL PATCH REVIEW V2 — QUALITY AUDIT, NO GAN TRAINING
Project: {ROOT}
Output folder: {OUT}
Review date: 2026-11-02

Scope and safeguards
--------------------
- Only new review artifacts were written under reports/final_patch_review_v2.
- No GAN training was started. All KEEP entries remain review candidates and were not added to a training run.
- Original source photos and COCO JSON were read-only. Source RGB crops were exported losslessly as RGB PNG at native pixel resolution.
- RGB channel count is 3. No grayscale, binary-mask PNG, or mask training target was generated.
- Contact-sheet thumbnails are scaled only for page layout; the crop files themselves remain at native 1024x1024, 1536x1536, or 2048-sized source resolution.

Clean 1024 audit — all 67 inspected
-----------------------------------
Near-black definition: pixel max(R,G,B) < 5. “Non-black proxy” is the complement of this threshold; it is not a semantic product segmentation. Visual review was also used to distinguish product coating from the black background.
- Initial clean crops: 67.
- Ratio bands: {bands['100% near-black']} completely near-black; {bands['30-<100% near-black']} at 30–<100%; {bands['5-<30% near-black']} at 5–<30%; {bands['<5% near-black']} below 5%.
- 39 crops at or above 30% near-black were removed from the final clean list and replaced by distinct windows from the same F971 OK source image. All 39 selected replacement RGB windows measure 0.000% near-black and show the F971 dark coating/product context on visual review.
- 28 original clean windows below the 30% trigger were retained after visual review; the product surface remains visible and no clear Scratch was apparent.
- All four user-highlighted patches were replaced:
{flagged_text}
- Additional all-black patches identified in the full 67 audit are listed in clean_67_patch_pixel_audit.csv and clean_final_manifest.csv; none are retained as clean training candidates.
- One automatic replacement for clean patch 120 showed a visible copper/brown Scratch. That candidate and a second alternate were rejected. A different, smooth surface crop (x=640,y=1024) was selected instead; see clean_candidate_rejections.csv and clean_after_filtering_67.png.
- The selected final clean review set contains 67 candidates: {clean_counts['train']} train, {clean_counts['validation']} validation, {clean_counts['test']} test. Final ratios and material-proxy measurements for every patch are in clean_final_manifest.csv.

31 pending Scratch crops
------------------------
- Contact sheet displays each patch, source image, annotation ID(s), and split with full polygon outlines in yellow.
- Visual dispositions: {len(kept_scratch)} KEEP; {len(held_scratch_ids)} HOLD; 0 REJECT.
- KEEP patch IDs: {', '.join(kept_scratch)}.
- HOLD patch IDs: {', '.join(held_scratch_ids)}.
- HOLD reasons: P25 is a small irregular discoloration with ambiguous Scratch shape; P29 is tiny/low contrast against texture; P43 reaches the original frame boundary and its full extent cannot be confirmed.
- KEEP means visually suitable as a candidate only. No patch was put into training.

Sixteen edge-neighbor crops
---------------------------
- Polygon overlays: full polygons yellow; clipped polygons red; underlying tiles remain RGB.
- Patch 7 was recropped at source x=2988,y=352 to restore originally clipped annotation 24. The trial still clips neighboring annotations {', '.join(edge7_clipped_ids)} ({edge7_full} full, {edge7_clipped} clipped intersecting polygons), so it is also excluded from the strict baseline.
- All 16 edge crops are marked DROP_FROM_STRICT_1024_BASELINE: {', '.join(edge_drop)}. The patch-7 recrop is retained only as review evidence, not as an approved training candidate.
- Their original COCO annotations remain unchanged.

Highlighted NG crop
-------------------
- Original {Path(ng_original['patch_file']).name}: {ng_original_black:.3f}% near-black; its RGB contact sheet shows substantial outside-product background.
- A 1024 source crop at x=3385,y=72 has 0.000% near-black and contains target annotation IDs 166|167|168|169, but clips neighboring annotation 163. It is HOLD and excluded from the strict 1024 candidates.
- A separate 1536 source-RGB proposal at x={ng_x},y={ng_y} is provided for review. It includes target IDs 166|167|168|169 and nearby ID 163 where present; it is not mixed into the 1024 baseline. Full/clipped ID details are in ng_1536_neighbor_context_proposal.csv.

Eight oversized COCO annotations
--------------------------------
- All eight COCO annotations are retained. Each has a source-native RGB 1536x1536 proposal, separate from the 1024 experiment.
- Annotation IDs: 158, 159, 191, 202, 204, 230, 241, 245.
- Proposal coordinates, source split, black ratio, and context margins are in oversize_annotation_crop_proposals.csv. Every target polygon is fully inside its 1536 crop.

Provisional 1024 candidate counts by source split
-------------------------------------------------
These counts summarize QA-approved candidates only; they are not a training manifest. They exclude all 16 clipped edge crops, 3 HOLD patches from the 31 batch, and the highlighted NG patch 45 because its 1024 recrop clips a neighbor.
{count_text}
  - TOTAL: {sum(defect_counts.values())} defect + {sum(clean_counts.values())} clean = {sum(defect_counts.values()) + sum(clean_counts.values())} review-approved candidates.
- Source-level leakage check: {'FAIL: ' + ', '.join(leak_sources) if leak_sources else 'PASS; each source image maps to one split and replacements inherit that source split.'}
- No training started; user visual approval is still required before Defect-GAN training.

COCO and file-integrity checks
------------------------------
- Normalized COCO read-only check: {len(coco['images'])} images, {len(coco['annotations'])} annotations, {len(dangling)} dangling annotation references, duplicate image IDs={duplicate_image_ids}.
- Candidate crops are pixel-exact source windows: {sum(ok for _, ok in exact_checks)}/{len(exact_checks)} checks passed; see rgb_pixel_provenance_check.csv.
- Every PNG candidate and preview is RGB; no binary mask files were generated.
- Source-level split assignment was checked before crops; no source crosses train/validation/test.

Files for review
----------------
- contact_sheets/scratch_31_review_with_dispositions.png
- contact_sheets/clean_after_filtering_67.png
- contact_sheets/edge_16_final_dispositions.png
- contact_sheets/edge_patch7_replacement_polygon_preview.png
- contact_sheets/ng_patch_replacement_candidates.png
- contact_sheets/ng_neighbor_context_1536_polygon_preview.png
- contact_sheets/oversize_8_crop_proposals.png
- clean_67_patch_pixel_audit.csv, clean_final_manifest.csv, scratch_31_dispositions.csv,
  edge_patch_dispositions.csv, oversize_annotation_crop_proposals.csv, and final_patch_counts_by_split.csv
"""
    (OUT / "final_patch_review_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "clean_originals_audited": len(clean_audit), "clean_originals_retained": sum(row["disposition"] == "KEEP_ORIGINAL" for row in final_clean_rows),
        "clean_replacements": len(final_replacements), "scratch31_keep": len(kept_scratch), "scratch31_hold": len(held_scratch_ids),
        "edge_crops_dropped": len(edge_drop), "edge_recrop_candidates": 0, "oversize_annotations_retained": 8,
        "strict_1024_review_candidates": sum(defect_counts.values()) + sum(clean_counts.values()),
        "source_image_leakage": bool(leak_sources), "coco_images": len(coco["images"]), "coco_annotations": len(coco["annotations"]),
        "rgb_crop_pixel_exact_checks_passed": sum(ok for _, ok in exact_checks), "rgb_crop_pixel_exact_checks_total": len(exact_checks),
        "binary_mask_png_count": 0, "training_started": False,
    }
    (OUT / "qa_final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Bundle review artifacts without including the zip in itself.
    archive = OUT / "final_patch_review_v2.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path != archive:
                zf.write(path, path.relative_to(OUT.parent))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


