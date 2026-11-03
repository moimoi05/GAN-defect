# HƯỚNG DẪN TRIỂN KHAI VÀ HUẤN LUYỆN DEFECT-GAN TRÊN MÁY GPU CÔNG TY

Tài liệu này hướng dẫn chi tiết quy trình chuyển giao mã nguồn, đóng gói Docker, kiểm tra phần cứng và thực thi quá trình huấn luyện mô hình Defect-GAN trên máy chủ GPU NVIDIA tại công ty.

---

## 1. QUY TRÌNH CHUYỂN GIAO SANG MÁY CÔNG TY

Có 2 phương thức chuyển giao tùy thuộc vào kết nối mạng của máy công ty:

### Phương thức 1: Xuất Docker Image dạng file nén (Khuyên dùng nếu máy công ty không có Internet)
Trên máy phát triển (Windows có Docker):
```powershell
# 1. Build image
docker build -t defectgan:latest .

# 2. Xuất image thành file .tar
docker save defectgan:latest -o defectgan_image.tar

# 3. Nén mã nguồn và data (chỉ cần training/manifests, src, configs, scripts)
# Không cần nén outputs/ hay cache
```
Trên máy công ty (Linux GPU):
```bash
# 1. Tải image vào Docker
docker load -i defectgan_image.tar

# 2. Kiểm tra image đã có trong danh sách
docker images | grep defectgan
```

### Phương thức 2: Build trực tiếp trên máy công ty (Nếu máy công ty có kết nối mạng)
```bash
# Clone hoặc copy thư mục GAN_Project sang máy công ty
cd /path/to/GAN_Project

# Build Docker image
docker build -t defectgan:latest .
```

---

## 2. KIỂM TRA PHẦN CỨNG GPU TRƯỚC KHI HUẤN LUYỆN

Trước khi cấu hình tham số, bắt buộc phải chạy script chẩn đoán phần cứng để biết chính xác model GPU, dung lượng VRAM thực tế và phiên bản driver.

### 2.1. Kiểm tra trên Host Linux
```bash
nvidia-smi
```
Ghi nhận:
- GPU Model (ví dụ: RTX 3090, RTX 4090, A5000, A100, T4, RTX 3060).
- Driver Version (Yêu cầu >= 530 cho CUDA 12.1).
- CUDA Version hiển thị ở góc trên bên phải.

### 2.2. Kiểm tra bên trong Docker Container
```bash
docker run --rm --gpus all defectgan:latest check-env
```
Script sẽ tự động phân loại VRAM và in ra khuyến nghị cấu hình phù hợp.

---

## 3. CẤU HÌNH THAM SỐ THEO DUNG LƯỢNG VRAM THỰC TẾ

Chỉnh sửa file `configs/defectgan_1024.yaml` (hoặc tạo file config mới tương ứng) dựa trên kết quả kiểm tra VRAM:

| Phân khúc VRAM | Model GPU tiêu biểu | Khả năng chạy 1024×1024 | `batch_size` | `gradient_accumulation_steps` | `mixed_precision` | `num_workers` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **>= 24 GB** | RTX 3090, 4090, A5000, A6000, A100 | **Rất tốt** | `1` hoặc `2` | `2` hoặc `4` (effective batch=4) | `fp16` / `bf16` | `4` |
| **16 GB** | RTX 4080, V100, T4 16GB, RTX 3080 16G | **Tốt** | `1` | `4` (effective batch=4) | `fp16` (bắt buộc) | `2` |
| **10 – 12 GB** | RTX 3060 12GB, RTX 4070, RTX 3080 10G | **Nguy cơ OOM** | `1` | `4` | `fp16` (bắt buộc) | `2` |
| **< 10 GB** | GPU laptop hoặc thẻ 8GB (RTX 3070 8G) | **Không khả thi 1024** | *Chuyển sang cấu hình 512×512* | `4` | `fp16` | `0` hoặc `2` |

> **Cảnh báo VRAM 1024x1024:** Quá trình huấn luyện GAN với 2 Generator (Defacement + Restoration) và 2 Discriminator Multi-Scale, kèm Cycle-Consistency backpropagation trên ảnh 1024×1024 tiêu tốn khoảng **14 – 18 GB VRAM** ở chế độ FP16. Nếu máy công ty có VRAM <= 12GB và gặp lỗi CUDA Out-Of-Memory, xem mục 7 để xử lý.

---

## 4. KHỞI CHẠY HUẤN LUYỆN TRÊN GPU

### 4.1. Sử dụng Docker Compose (Khuyên dùng)
1. Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```
2. Cập nhật đường dẫn trong `.env`:
```env
DATA_DIR=/absolute/path/to/GAN_Project
OUTPUT_DIR=/absolute/path/to/GAN_Project/outputs
CUDA_VISIBLE_DEVICES=all
```
3. Khởi chạy container huấn luyện ở chế độ nền:
```bash
docker compose up -d defectgan
```
4. Theo dõi log:
```bash
docker compose logs -f defectgan
```

### 4.2. Sử dụng lệnh `docker run` trực tiếp
```bash
docker run --gpus all --ipc=host \
  --name defectgan-training \
  -v "/path/to/GAN_Project:/workspace/data:ro" \
  -v "/path/to/GAN_Project/outputs:/workspace/outputs" \
  -d defectgan:latest train --config configs/defectgan_1024.yaml
```

> **Lưu ý an toàn dữ liệu:** Thư mục dataset `/workspace/data` luôn được mount ở chế độ **read-only (`:ro`)**, bảo đảm container không thể vô tình sửa đổi hay xóa ảnh nguồn gốc.

---

## 5. THEO DÕI HUẤN LUYỆN & KIỂM TRA KẾT QUẢ

### 5.1. Xem Loss Log
File log được ghi liên tục tại:
`outputs/f971_defectgan_1024_baseline/train.log`

Xem trực tiếp theo thời gian thực:
```bash
tail -f outputs/f971_defectgan_1024_baseline/train.log
```
Các chỉ số cần chú ý:
- `D_loss`: Giảm dần và dao động ổn định quanh mức 1.0 – 2.5 (LSGAN). Nếu tụt về 0.0 là Discriminator áp đảo Generator.
- `G_loss`: Giảm dần theo số epoch.
- `g_cyc_normal`: L1 cycle loss của ảnh OK. Chỉ số này giảm chứng tỏ cấu trúc nền được khôi phục tốt.
- `g_mask_defect`: Diện tích trung bình của mask $M$. Phải duy trì ở mức nhỏ (< 0.20), chứng tỏ vết xước chỉ chiếm một vùng nhỏ và nền xung quanh không bị thay đổi.

### 5.2. Xem ảnh RGB mẫu được sinh ra định kỳ
Sau mỗi 5 epoch (hoặc cấu hình trong `sample_interval`), Evaluator tự động lưu ảnh mẫu vào:
`outputs/f971_defectgan_1024_baseline/samples/epoch_XXXX/`

Mỗi mẫu gồm 7 ảnh RGB 1024×1024:
1. `sample_0_01_normal_input.png`: Ảnh F971 OK ban đầu.
2. `sample_0_02_fake_defect.png`: **Ảnh sinh lỗi Scratch tổng hợp.**
3. `sample_0_03_defect_mask.png`: Visual mask phân bố vị trí lỗi nội bộ.
4. `sample_0_04_recon_normal.png`: Ảnh phục hồi ngược lại từ ảnh sinh lỗi.
5. `sample_0_05_defect_input.png`: Ảnh F971 NG lỗi thật đầu vào.
6. `sample_0_06_fake_normal.png`: Ảnh sau khi xóa lỗi (Restoration).
7. `sample_0_07_recon_defect.png`: Ảnh tái tạo lại chu trình defect.

---

## 6. KHÔI PHỤC HUẤN LUYỆN (RESUME CHECKPOINT)

Nếu tiến trình huấn luyện bị gián đoạn (mất điện, bảo trì), hệ thống tự động lưu file `latest.pth` tại:
`outputs/f971_defectgan_1024_baseline/checkpoints/latest.pth`

Lệnh tiếp tục huấn luyện:
```bash
# Qua Docker run:
docker run --gpus all --ipc=host \
  -v "/path/to/GAN_Project:/workspace/data:ro" \
  -v "/path/to/GAN_Project/outputs:/workspace/outputs" \
  defectgan:latest train \
    --config configs/defectgan_1024.yaml \
    --resume outputs/f971_defectgan_1024_baseline/checkpoints/latest.pth

# Hoặc qua docker-compose (chỉnh sửa command trong docker-compose.yml):
command: ["train", "--config", "configs/defectgan_1024.yaml", "--resume", "/workspace/outputs/f971_defectgan_1024_baseline/checkpoints/latest.pth"]
```

---

## 7. CHẠY INFERENCE TRÊN ẢNH F971 VÀ S942

### 7.1. Sinh lỗi Scratch trên F971 OK
```bash
docker run --gpus all --rm \
  -v "/path/to/GAN_Project:/workspace/data:ro" \
  -v "/path/to/GAN_Project/outputs:/workspace/outputs" \
  defectgan:latest infer \
    --checkpoint /workspace/outputs/f971_defectgan_1024_baseline/checkpoints/best.pth \
    --input-dir /workspace/data/F971/OK/images \
    --output-dir /workspace/outputs/inference_f971_scratch \
    --mode defacement \
    --save-mask
```

### 7.2. Thử nghiệm sinh lỗi trên mã sản phẩm mới S942 OK
Ảnh S942 OK có kích thước rất lớn (11004 × 5599). Script `src/inference.py` tích hợp sẵn cơ chế **Sliding-Window Hann Blending**:
```bash
docker run --gpus all --rm \
  -v "/path/to/GAN_Project:/workspace/data:ro" \
  -v "/path/to/GAN_Project/outputs:/workspace/outputs" \
  defectgan:latest infer \
    --checkpoint /workspace/outputs/f971_defectgan_1024_baseline/checkpoints/best.pth \
    --input-dir /workspace/data/S942/OK/images \
    --output-dir /workspace/outputs/inference_s942_scratch \
    --mode defacement \
    --patch-size 1024 \
    --stride 768 \
    --save-mask
```

> **QUAN TRỌNG VỀ ĐÁNH GIÁ S942:**
> Do S942 là mã sản phẩm khác F971 (khác bề mặt, khác góc chụp và kích thước), việc script chạy thành công và tạo ra file ảnh **KHÔNG ĐỒNG NGHĨA** với việc chất lượng sinh lỗi trên S942 đã đạt yêu cầu. Đội ngũ kỹ sư cần kiểm tra trực quan hiện tượng Domain Shift (màu sắc lớp phủ, vân gia công) để quyết định có cần fine-tune thêm với ảnh S942 hay không.

---

## 8. XỬ LÝ SỰ CỐ (TROUBLESHOOTING)

### 8.1. Lỗi CUDA Out-Of-Memory (OOM)
Nếu GPU máy công ty có VRAM < 16GB và xuất hiện thông báo `torch.cuda.OutOfMemoryError`:
1. Giảm kích thước batch xuống `1` (đã là mặc định).
2. Tăng `gradient_accumulation_steps: 4` hoặc `8`.
3. Bật mixed precision: `mixed_precision: fp16`.
4. Nếu vẫn tràn bộ nhớ, tạo cấu hình patch 512×512 (sử dụng thư mục `experiments/patches_512x512`).

### 8.2. Lỗi `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`
Lỗi này xuất hiện khi máy công ty chưa cài đặt **NVIDIA Container Toolkit**:
```bash
# Hướng dẫn cài đặt trên Ubuntu/Debian:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 8.3. Lỗi NaN hoặc Inf trong Loss
Mã nguồn đã tích hợp hàm `check_tensor_valid()` để bắt lỗi NaN/Inf ngay tại iteration xuất hiện.
Nếu loss xuất hiện NaN:
1. Giảm learning rate của Generator xuống `0.00005` và Discriminator xuống `0.0002`.
2. Kiểm tra xem có ảnh đầu vào nào bị lỗi dữ liệu bằng `python scripts/validate_dataset.py`.
