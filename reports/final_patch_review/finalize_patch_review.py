"""Record the completed visual patch review and emit its text report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "final_patch_review"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_rgb_native_copies(samples: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in samples:
        size = int(row["experiment"].split("x")[0])
        source = ROOT / "experiments" / f"patches_{size}x{size}" / row["patch_name"]
        if not source.is_file():
            source = ROOT / "experiments" / f"patches_{size}x{size}" / "patch_manifest_images" / row["patch_name"]
        # Manifest paths are resolved below using the original patch ID and experiment.
        manifest_path = ROOT / "experiments" / f"patches_{size}x{size}" / "patch_manifest.csv"
        manifest_rows = read_csv(manifest_path)
        source_row = next((item for item in manifest_rows if item["patch_id"] == row["patch_id"]), None)
        if source_row is None:
            errors.append(f"Missing source manifest row: {row['experiment']} patch_id={row['patch_id']}")
            continue
        source = ROOT / "experiments" / f"patches_{size}x{size}" / source_row["patch_file"]
        native = OUT / row["native_copy"]
        if not source.is_file() or not native.is_file():
            errors.append(f"Missing RGB patch source/copy: {row['patch_name']}")
            continue
        with Image.open(source) as image:
            if image.mode != "RGB" or image.size != (size, size):
                errors.append(f"Source is not native RGB {size}x{size}: {row['patch_name']} {image.mode} {image.size}")
        with Image.open(native) as image:
            if image.mode != "RGB" or image.size != (size, size):
                errors.append(f"Review copy is not native RGB {size}x{size}: {row['patch_name']} {image.mode} {image.size}")
        if sha256(source) != sha256(native):
            errors.append(f"Review copy differs from patch source: {row['patch_name']}")
    return errors


def verify_native_roi_pixels(roi_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in roi_rows:
        source_path = ROOT / "F971" / "NG" / "images" / row["source_image"]
        output_path = OUT / "native_scratch_crops" / row["rgb_file"]
        if not source_path.is_file() or not output_path.is_file():
            errors.append(f"Missing source/output ROI: {row['source_image']} / {row['rgb_file']}")
            continue
        left = int(row["roi_x"])
        top = int(row["roi_y"])
        right = left + int(row["roi_width"])
        bottom = top + int(row["roi_height"])
        with Image.open(source_path) as source, Image.open(output_path) as output:
            if source.mode != "RGB":
                errors.append(f"Source ROI image is not RGB: {source_path.name} ({source.mode})")
                continue
            expected = source.crop((left, top, right, bottom))
            if expected.size != output.size or expected.mode != output.mode or expected.tobytes() != output.tobytes():
                errors.append(f"Raw ROI pixels differ from original source crop: {row['rgb_file']}")
    return errors


def main() -> None:
    summary = json.loads((OUT / "review_build_summary.json").read_text(encoding="utf-8"))
    samples = read_csv(OUT / "review_sample_manifest.csv")
    clean_rows = read_csv(OUT / "clean_patch_visual_audit.csv")
    edge_rows = read_csv(OUT / "edge_case_review.csv")
    all_1024_rows = read_csv(ROOT / "experiments" / "patches_1024x1024" / "patch_manifest.csv")

    # All 67 clean 1024 patches were inspected at native RGB resolution.
    for row in clean_rows:
        row["visual_status"] = "PASS_NO_OBVIOUS_SCRATCH"
        row["recommended_action"] = "KEEP_AS_CLEAN_CANDIDATE"
        row["notes"] = (
            "F971 OK RGB, no COCO annotation, visually inspected at native resolution; "
            "no Scratch matching the NG examples is apparent. Fine machining texture remains."
        )
    write_csv(OUT / "clean_patch_visual_audit.csv", clean_rows)

    # Every edge entry has a red clipped segment in the RGB overlay and is excluded
    # from a strict baseline to avoid partially represented neighboring defects.
    for row in edge_rows:
        row["visual_review_status"] = "FAIL_PARTIAL_NEIGHBOR_SCRATCH_VISIBLE"
        row["recommended_action"] = "DROP_FROM_STRICT_1024_BASELINE"
        row["review_notes"] = (
            "Red overlay marks Scratch polygon portions cut by the patch boundary; "
            "yellow polygons are fully represented annotations. Re-crop from source only if the full shape fits."
        )
    write_csv(OUT / "edge_case_review.csv", edge_rows)

    for row in samples:
        if row["experiment"] == "1024x1024" and row["label"] == "defect":
            row["visual_review_status"] = "PASS_VISIBLE_SCRATCH"
            row["recommended_action"] = "KEEP_1024_BASELINE_CANDIDATE"
            row["review_notes"] = "Representative native RGB patch; Scratch is visible and its target polygon is fully inside the crop."
        elif row["experiment"] == "1024x1024" and row["label"] == "clean":
            row["visual_review_status"] = "PASS_NO_OBVIOUS_SCRATCH"
            row["recommended_action"] = "KEEP_AS_CLEAN_CANDIDATE"
            row["review_notes"] = "No Scratch matching NG examples is apparent; also included in the 67-patch clean audit."
        elif row["experiment"] == "512x512" and row["label"] == "defect":
            row["visual_review_status"] = "PASS_VISIBLE_SCRATCH"
            row["recommended_action"] = "KEEP_FOR_512_COMPARISON_ONLY"
            row["review_notes"] = "Native RGB sample for a separate 512x512 comparison; do not mix with the 1024 baseline."
        else:
            row["visual_review_status"] = "PASS_NO_OBVIOUS_SCRATCH"
            row["recommended_action"] = "KEEP_FOR_512_COMPARISON_ONLY"
            row["review_notes"] = "Clean-looking RGB comparison sample; do not mix with the 1024 baseline."
    write_csv(OUT / "review_sample_manifest.csv", samples)

    selected_positive_ids = {
        row["patch_id"] for row in samples
        if row["experiment"] == "1024x1024" and row["label"] == "defect"
    }
    edge_patch_ids = {row["patch_id"] for row in edge_rows}
    unreviewed_positive_rows = []
    for row in all_1024_rows:
        if row["label"] != "defect" or row["patch_id"] in selected_positive_ids | edge_patch_ids:
            continue
        unreviewed_positive_rows.append(
            {
                "patch_id": row["patch_id"],
                "patch_name": Path(row["patch_file"]).name,
                "source_image": row["source_image"],
                "split": row["split"],
                "annotation_ids": row["annotation_ids"],
                "annotation_count": row["annotation_count"],
                "visual_review_status": "NOT_INDIVIDUALLY_REVIEWED",
                "recommended_action": "HOLD_PENDING_VISUAL_REVIEW",
                "notes": "Scratch target is represented in the patch manifest; patch was outside the requested representative sample and needs visual review before training.",
            }
        )
    with (OUT / "unreviewed_positive_1024.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unreviewed_positive_rows[0]))
        writer.writeheader()
        writer.writerows(unreviewed_positive_rows)

    integrity_errors = verify_rgb_native_copies(samples)
    all_pngs = sorted(OUT.rglob("*.png"))
    mask_named_pngs = [path for path in all_pngs if "mask" in path.name.casefold()]
    non_rgb_pngs: list[str] = []
    for path in all_pngs:
        with Image.open(path) as image:
            if image.mode != "RGB":
                non_rgb_pngs.append(f"{path.relative_to(OUT)} ({image.mode})")
    roi_rows = read_csv(OUT / "native_scratch_crop_manifest.csv")
    roi_errors = verify_native_roi_pixels(roi_rows)
    for row in roi_rows:
        for column, expected in (("rgb_file", "RGB"), ("overlay_file", "RGB")):
            path = OUT / "native_scratch_crops" / row[column]
            with Image.open(path) as image:
                if image.mode != expected:
                    roi_errors.append(f"{path.name}: mode={image.mode}")
                if image.size != (int(row["roi_width"]), int(row["roi_height"])):
                    roi_errors.append(f"{path.name}: size={image.size}")

    if integrity_errors or mask_named_pngs or non_rgb_pngs or roi_errors:
        raise RuntimeError(
            "Review artifact integrity failed:\n"
            + "\n".join(integrity_errors + [f"Mask-named PNG: {p.name}" for p in mask_named_pngs]
                      + [f"Non-RGB PNG: {item}" for item in non_rgb_pngs] + roi_errors)
        )

    scratch_rows = [row for row in samples if row["experiment"] == "1024x1024" and row["label"] == "defect"]
    clean_sample_rows = [row for row in samples if row["experiment"] == "1024x1024" and row["label"] == "clean"]
    compare_rows = [row for row in samples if row["experiment"] == "512x512"]
    edge_ids = ", ".join(row["patch_id"] for row in edge_rows)
    lines = [
        "FINAL PATCH REVIEW REPORT — Defect-GAN (visual QA only)",
        "=" * 64,
        "Workspace: D:\\RTC\\RD\\GAN_Project",
        "Output: D:\\RTC\\RD\\GAN_Project\\reports\\final_patch_review",
        "Review scope: no GAN training or inference was run.",
        "",
        "1. Số lượng patch và kết quả kiểm tra",
        f"- Bộ 1024x1024 hiện có: 134 patch (67 có Scratch, 67 clean), tỷ lệ 50/50.",
        f"- Đã kiểm tra trực quan ở 1024: {len(scratch_rows)} patch Scratch đại diện + {len(edge_rows)} patch có Scratch lân cận bị cắt + {len(clean_rows)} patch clean = {len(scratch_rows) + len(edge_rows) + len(clean_rows)} patch duy nhất.",
        f"- Bộ 512x512 hiện có: 142 patch (71 có Scratch, 71 clean); đã kiểm tra mẫu so sánh {len(compare_rows)} patch (5 Scratch, 5 clean).",
        f"- Tổng số patch duy nhất đã kiểm tra: {len(scratch_rows) + len(edge_rows) + len(clean_rows) + len(compare_rows)}; 10 patch clean trong contact sheet chính nằm trong 67 patch clean đã kiểm tra, không tính trùng.",
        f"- Kết quả clean: {len(clean_rows)}/67 PASS_NO_OBVIOUS_SCRATCH; không thấy Scratch rõ tương tự mẫu NG. Vân gia công mảnh trên bề mặt kim loại được giữ nguyên và không bị tự động gán lỗi.",
        f"- Kết quả patch Scratch đại diện 1024: {len(scratch_rows)}/{len(scratch_rows)} PASS; polygon mục tiêu nằm trọn trong crop.",
        f"- Trường hợp cần loại khỏi baseline nghiêm ngặt: {len(edge_rows)}/{len(edge_rows)} patch có polygon Scratch lân cận bị cắt qua mép (đoạn đỏ trong overlay). Patch chứa Scratch được gán nhãn đầy đủ vẫn hiện diện, nhưng có thêm lỗi không trọn hình.",
        f"- Mã patch edge bị cắt: {edge_ids}.",
        f"- Mẫu 512: {sum(row['label'] == 'defect' for row in compare_rows)} Scratch + {sum(row['label'] == 'clean' for row in compare_rows)} clean, đạt cho so sánh thị giác; chỉ dùng trong experiment 512 riêng.",
        "- 31 patch Scratch 1024 không nằm trong 20 mẫu đại diện và chưa được kiểm tra từng patch.",
        f"- Khuyến nghị cho 31 patch này: HOLD_PENDING_VISUAL_REVIEW; danh sách chi tiết ở unreviewed_positive_1024.csv.",
        "",
        "2. Kết quả đạt yêu cầu",
        "- 20 patch Scratch 1024 đại diện cho các dải chiều dài khác nhau: giữ làm ứng viên baseline; xem review_sample_manifest.csv để biết tên, ảnh nguồn, split, số annotation và chiều dài lớn nhất.",
        "- 67 patch clean 1024: giữ làm ứng viên clean sau kiểm tra trực quan; không phát hiện Scratch rõ trên preview RGB. Đây là đánh giá bằng mắt, không thay thế duyệt chuyên môn cuối cùng.",
        "- 10 patch 512: giữ trong experiment so sánh 512 riêng, không gộp vào baseline 1024.",
        f"- {len(roi_rows)} ROI Scratch được trích ở tọa độ và độ phân giải pixel gốc (kích thước thay đổi theo bbox + context); file *_rgb.png là ảnh RGB nguyên bản, *_polygon_overlay.png chỉ là bản xem có nét polygon.",
        "",
        "3. Trường hợp lỗi và quyết định từng patch",
        "- 16/16 edge cases: DROP_FROM_STRICT_1024_BASELINE. Nguyên nhân là một hoặc nhiều Scratch lân cận chỉ còn một phần trong crop; ID polygon bị ảnh hưởng và số lượng được ghi riêng trong edge_case_review.csv.",
        "- Đề xuất xử lý: không đưa 16 patch này vào tập train baseline ở trạng thái hiện tại. Nếu muốn tận dụng, tạo crop lại từ ảnh nguồn sao cho mọi Scratch trong vùng đều trọn hình; giữ nguyên RGB và không resize.",
        "- 8 annotation không thể vừa crop 1024 ở kích thước pixel gốc đã được ghi trong crop_exclusions_1024.csv; không xuất hiện dưới dạng target đầy đủ trong baseline 1024. Cần patch lớn hơn hoặc giữ ngoài experiment này.",
        "",
        "4. Kiểm tra RGB và tính toàn vẹn artifact",
        f"- Patch mẫu được lưu ở kích thước gốc 1024x1024 hoặc 512x512, mode RGB 3 kênh; SHA-256 của bản copy trong native_samples trùng với file patch gốc ({len(samples)} mẫu).",
        "- Sheet ghép ảnh dùng từng tile native-size, thêm footer riêng để ghi tên patch/ảnh nguồn; không resize pixel ảnh tile.",
        "- Overlay polygon được lưu thành preview RGB riêng: polygon đầy đủ màu vàng, phần polygon bị cắt màu đỏ. File patch RGB gốc không bị vẽ đè.",
        f"- ROI RGB và overlay ROI giữ cùng kích thước crop gốc; {len(all_pngs)} PNG trong thư mục review đều RGB; không tạo binary mask PNG.",
        "- Ảnh nguồn, coco_original.json, dataset làm việc và patch gốc chỉ được đọc; đầu ra mới nằm trong reports/final_patch_review.",
        "",
        "5. Thống kê Scratch trước khi crop",
        "- 245 annotation trên 49 ảnh F971 NG.",
        "- Chiều dài bbox: min 8 px; P25 60.04; median 156.35; P75 333; P90 608.18; P95 830; max 1385 px.",
        "- Chiều rộng bbox: min 1 px; P25 35.92; median 71.22; P75 142.52; P90 208.70; P95 269; max 407 px.",
        "- Có 8 annotation dài hơn khả năng bao trọn của patch 1024; kích thước và bbox đầy đủ nằm trong reports/scratch_bbox_statistics.csv.",
        "",
        "6. Kết luận baseline 1024 và đề xuất tài nguyên",
        "- 1024x1024 là baseline được ưu tiên như quyết định của người dùng. Sau khi loại 16 patch có Scratch lân cận bị cắt, còn 51 crop dương tính về mặt manifest + 67 clean = tối đa 118 crop. Hiện 20/51 crop dương tính đã được xem trực quan; 31 crop còn lại ở trạng thái HOLD_PENDING_VISUAL_REVIEW và không đưa vào train trước khi kiểm tra. Còn 8 annotation quá lớn chưa được đại diện đầy đủ.",
        "- GPU khởi điểm đề xuất: NVIDIA có ít nhất 24 GB VRAM; 1024x1024 dùng batch size 1, AMP, gradient accumulation để đạt effective batch 2–4. GPU 48 GB cho thêm dư địa bộ nhớ. Đây là ước lượng để profile sau khi được duyệt, chưa train.",
        "- Kiến trúc khởi điểm: Generator encoder-decoder RGB 3 kênh, điều kiện polygon/spatial tensor nội bộ (SPADE), noise injection cho texture, foreground + alpha nội bộ để composite lên RGB; PatchGAN discriminator và cycle/reconstruction path. Output cuối giữ 3 kênh RGB.",
        "- Chỉ bắt đầu train sau khi người dùng duyệt patch. 512x512 và 1024x1024 tiếp tục ở hai experiment tách biệt.",
        "",
        "7. Danh sách từng trường hợp",
        "A. 20 patch Scratch 1024 (đề xuất KEEP)",
    ]
    for row in scratch_rows:
        lines.append(
            f"- KEEP | {row['patch_name']} | nguồn {row['source_image']} | split {row['split']} | "
            f"annotations {row['annotation_count']} | max length {row['max_scratch_length_px']} px"
        )
    lines += ["", "B. 67 patch clean 1024 (đề xuất KEEP_AS_CLEAN_CANDIDATE)"]
    for row in clean_rows:
        lines.append(f"- KEEP | {row['patch_name']} | nguồn {row['source_image']} | split {row['split']} | no COCO annotations; no obvious Scratch in visual review")
    lines += ["", "C. 31 patch Scratch chưa xem từng ảnh (đề xuất HOLD)"]
    for row in unreviewed_positive_rows:
        lines.append(
            f"- HOLD | patch_id {row['patch_id']} | {row['patch_name']} | nguồn {row['source_image']} | "
            f"split {row['split']} | annotations {row['annotation_count']}"
        )
    lines += ["", "D. 16 patch có Scratch lân cận bị cắt (đề xuất DROP khỏi strict baseline)"]
    for row in edge_rows:
        lines.append(
            f"- DROP | patch_id {row['patch_id']} | {row['patch_name']} | nguồn {row['source_image']} | "
            f"split {row['split']} | clipped polygon IDs {row['partial_neighbor_annotation_ids']} "
            f"({row['partial_neighbor_count']} trường hợp)"
        )
    lines += ["", "E. 10 patch so sánh 512 (đề xuất KEEP trong experiment riêng)"]
    for row in compare_rows:
        lines.append(
            f"- KEEP_512_ONLY | {row['patch_name']} | nguồn {row['source_image']} | "
            f"label {row['label']} | split {row['split']} | annotations {row['annotation_count']}"
        )
    lines += [
        "",
        "Artifact contact sheet:",
        "- contact_sheet_1024_20_scratch_10_clean.png",
        "- contact_sheet_1024_16_edge_scratch_overlay.png",
        "- contact_sheet_512_comparison_5_scratch_5_clean.png",
        "- clean_audit_sheets/clean_audit_01.png ... clean_audit_08.png",
        "- edge_detail_sheets/edge_detail_01.png ... edge_detail_sheets/edge_detail_04.png",
        "- native_scratch_crops/ (6 raw RGB crops plus separate polygon overlay previews)",
        "",
        "No model training was run.",
    ]
    (OUT / "final_patch_review_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary.update(
        {
            "unique patch files visually reviewed": len(scratch_rows) + len(edge_rows) + len(clean_rows) + len(compare_rows),
            "1024 clean visual pass": len(clean_rows),
            "1024 representative Scratch visual pass": len(scratch_rows),
            "1024 clipped-neighbor Scratch fail for strict baseline": len(edge_rows),
            "1024 positive patches held for individual visual review": len(unreviewed_positive_rows),
            "512 comparison samples visually reviewed": len(compare_rows),
            "RGB PNG outputs": len(all_pngs),
            "binary mask PNG outputs": 0,
            "native patch copy hash mismatches": 0,
            "native ROI pixel mismatches": 0,
        }
    )
    (OUT / "review_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"report written; reviewed unique patches={len(scratch_rows) + len(edge_rows) + len(clean_rows) + len(compare_rows)}; png RGB={len(all_pngs)}; integrity errors=0")


if __name__ == "__main__":
    main()
