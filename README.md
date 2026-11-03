# Defect-GAN RGB: F971 Scratch research

Dự án nghiên cứu học từ ảnh F971 OK và F971 NG Scratch không ghép cặp, sau đó đánh giá trên F971 và khảo sát chuyển giao sang S942 OK. Đầu ra mục tiêu là **ảnh RGB ba kênh**, không phải segmentation mask. Chưa có bằng chứng mô hình đã học Scratch hoặc chuyển giao thành công sang S942. Không chạy full training trên máy cá nhân.

## Dataset và bảo toàn dữ liệu

Baseline là 114 PNG RGB 1024×1024 native: train 41 OK/31 NG, validation 9 OK/8 NG, test 17 OK/8 NG. Giữ 39 clean replacements; loại P25, P29, P43 (HOLD), 16 edge-clipped patches, P45 và oversize patches. Không thay đổi split, nguồn, COCO, audit hoặc sinh patch mới. Các manifests nằm trong `training/manifests/`; chi tiết và kết quả thực tế xem `reports/data_integrity_review.md` và `reports/verification_report.md`.

Validator mặc định **chỉ đọc**. Trên Windows với Python 3.11 và dependencies phù hợp:

```powershell
py -3.11 scripts/validate_dataset.py --data-root D:/RTC/RD/GAN_Project
py -3.11 -m pytest tests -v
py -3.11 scripts/run_smoke_test.py
```

Các lệnh trên là hướng dẫn chạy lại, không phải tuyên bố kết quả. Test CPU dùng tensor nhỏ và smoke config resize ảnh trong bộ nhớ; không ghi lại patch, không chứng minh chất lượng sinh hoặc dung lượng VRAM 1024. Xem log trong báo cáo xác minh để biết từng kết quả thực tế. Torch 2.1.2 trong `requirements.txt` hỗ trợ Python 3.11, không dùng Python 3.12 với bộ pin này.

## Kiến trúc và giới hạn

`configs/defectgan_1024.yaml` chọn `defectgan_paper_baseline`: generator dùng chung cho defacement/restoration, một critic có auxiliary classifier, layer-wise RGB composition, SPADE, noise, image reconstruction và spatial distribution cycle/area losses. Paper §4.1 dùng WGAN-GP và 5 cập nhật D cho một cập nhật G. Thiết lập 1024, các chi tiết implementation, GP coefficient và lịch học trong project là adaptation; không tuyên bố reproduction 100%. Biến thể hai generator/hai discriminator cùng LSGAN được tách thành `defectgan_custom_legacy`. Xem [paper alignment](reports/paper_alignment_review.md) và [paper](https://arxiv.org/pdf/2103.15158).

Composition khuyến khích giữ nền, nhưng không bảo đảm tuyệt đối. `--roi` và `--protect` là ràng buộc compositing bổ sung của ứng dụng; vùng được bảo vệ phải do người dùng cung cấp. `--spatial-map` điều kiện hóa SPADE, khác với internal alpha sinh ra. `--save-alpha` chỉ lưu visualization debug, không phải ground truth hay ảnh NG RGB. Inference mặc định deterministic; `--noise stochastic --seed 42` chọn nhiễu có thể tái lập. Khác seed chỉ tạo khác biệt khi trọng số noise đã học có tác dụng.

## Docker và handover

Image sử dụng PyTorch 2.1.2/CUDA 12.1; code ở `/workspace`, dataset mount read-only `/workspace/data`, outputs writable `/workspace/outputs`. Dataset mount phải bao gồm cả patch candidates và báo cáo audit cần cho validator, không chỉ ảnh gốc.

```bash
docker build -t defectgan:latest .
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest validate-data
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest smoke-test
```

Kiểm tra trạng thái build/container thực tế trong `reports/verification_report.md`. Quy trình chẩn đoán và bounded GPU smoke trên máy công ty nằm trong [TRAINING_GUIDE.md](TRAINING_GUIDE.md) và [gpu_handover_checklist.md](reports/gpu_handover_checklist.md). Chưa chọn cấu hình VRAM hay chạy full training trước khi có kết quả GPU smoke.
