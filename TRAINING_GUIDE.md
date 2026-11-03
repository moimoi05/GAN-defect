# Hướng dẫn handover và GPU smoke

Mục tiêu bước tiếp theo là kiểm tra kỹ thuật có giới hạn, **không khởi chạy full training**. Đọc `reports/verification_report.md` để xác định những gì đã thực sự chạy. GPU, driver, VRAM và NVIDIA Container Toolkit của máy công ty hiện chưa được xác định. Không ước lượng batch size hoặc khẳng định 1024 vừa VRAM trước khi đo.

## 1. Chuẩn bị

Chuyển source và toàn bộ cây dữ liệu đã phê duyệt, gồm `training/manifests`, `experiments` và `reports/final_patch_review_v2`, giữ nguyên các đường dẫn tương đối. Nếu dùng image build sẵn, chuyển image bằng `docker save`/`docker load`; vẫn phải mount dữ liệu đầy đủ. Không sửa source images, COCO, audit, split hoặc tái tạo patch.

Trên Linux x86_64 có Docker và NVIDIA Container Toolkit, dùng placeholder đường dẫn tuyệt đối:

```bash
cd <ABS_PROJECT_ROOT>
mkdir -p <ABS_OUTPUT_ROOT>
docker build -t defectgan:latest .
nvidia-smi
```

Để lưu diagnostics lâu dài, mount output ở lệnh cuối:

```bash
docker run --rm --gpus all -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" \
  defectgan:latest check-env --output /workspace/outputs/gpu_diagnostics.json
```

Ghi lại tên GPU, count, total/free VRAM, driver, Torch/CUDA runtime, FP16/BF16 và phiên bản Docker. CUDA hiển thị ở `nvidia-smi` không phải kết quả đo VRAM training. Kiểm tra driver tương thích runtime CUDA 12.1 và toolkit trên máy thực tế.

## 2. Đọc dữ liệu và CPU smoke trong container

```bash
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" \
  -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest validate-data

docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" \
  -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest pytest -q

docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" \
  -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest smoke-test
```

Mỗi lệnh phải có exit code 0; xem log thay vì mặc định thành công. Smoke CPU dùng cấu hình model/tensor nhỏ, chỉ xác minh cơ chế và không chứng minh chất lượng ảnh hay hiệu năng GPU.

## 3. Bounded GPU smoke

```bash
docker run --rm --gpus all --ipc=host \
  -v "<ABS_DATA_ROOT>:/workspace/data:ro" \
  -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" \
  defectgan:latest gpu-smoke
```

Runner dùng `configs/defectgan_1024.yaml` và dừng sau thử nghiệm có giới hạn, không phải 200 epochs. Kiểm tra từng stage, exit code, loss/gradient hữu hạn, weight update của G và D, peak VRAM từ `dry_run_evidence.json`, checkpoint, resume và RGB inference. Nếu OOM hoặc lỗi, lưu log và trạng thái là FAIL/BLOCKED; không thay đổi độ phân giải hay kiến trúc baseline một cách âm thầm. Chỉ sau kết quả này mới cân nhắc microbatch, accumulation, AMP, workers, thời gian và checkpoint interval.

## 4. Checkpoint và inference

Checkpoint schema 2 chỉ cho resume từ epoch boundary an toàn. Emergency/mid-epoch checkpoint đánh dấu unsafe và không hỗ trợ exact resume; dùng `latest.pth` an toàn gần nhất. `best.pth` được chọn bằng validation reconstruction proxy, không phải bằng chứng Scratch realism. Chỉ load checkpoint từ nguồn đáng tin cậy; checkpoint baseline và legacy có kiến trúc/trọng số khác nhau.

Ví dụ inference sau khi đã có checkpoint phù hợp:

```bash
docker run --rm --gpus all -v "<ABS_DATA_ROOT>:/workspace/data:ro" \
  -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:latest infer \
  --checkpoint /workspace/outputs/<EXPERIMENT>/checkpoints/best.pth \
  --input /workspace/data/<INPUT_RGB.png> --output /workspace/outputs/result.png \
  --mode defacement --patch-size 1024 --stride 768 --device cuda \
  --noise stochastic --seed 42 --save-alpha
```

`--roi` và `--protect` nhận ảnh grayscale cùng kích thước đầu vào, và thực hiện bảo vệ bằng compositing bổ sung; `--spatial-map` là điều kiện SPADE riêng. Alpha debug không phải segmentation hoặc RGB NG. Cần xem README và `python -m src.inference --help` cho contract hiện tại. Sliding window kiểm thử logic CPU không chứng minh peak VRAM với ảnh S942 thực tế.

## 5. Đánh giá nghiên cứu sau khi huấn luyện được phê duyệt

Tập train có 31 NG: so sánh previews train/validation theo epoch, kiểm tra scratch lặp, nearest-neighbor đối chiếu ảnh NG train để phát hiện ghi nhớ, đánh giá riêng realism và bảo toàn nền/chữ/logo. Xem RGB input, RGB output, NG thật tham chiếu không ghép cặp, alpha debug và RGB difference. Reconstruction thấp không xác nhận realism; FID trên tập nhỏ rất bất ổn. Không chọn checkpoint hay hyperparameter bằng test set. S942 khác vật liệu, texture và bố cục; chỉ đánh giá transfer sau khi có mô hình và thử nghiệm thực tế.
