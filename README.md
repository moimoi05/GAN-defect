# Defect-GAN: High-Fidelity Scratch Synthesis Pipeline & Docker Deployment

Hệ thống sinh lỗi bề mặt công nghiệp (**Scratch**) độ trung thực cao sử dụng kiến trúc **Defect-GAN** (arXiv: 2103.15158) trên ảnh màu RGB 3 kênh.
Dự án được phát triển và kiểm định trên Windows (CPU), đóng gói Docker container sẵn sàng triển khai huấn luyện trên máy GPU NVIDIA tại công ty.

---

## 1. Mục Tiêu Dự Án

- **Bài toán:** Khắc phục tình trạng khan hiếm dữ liệu ảnh lỗi thực tế trong kiểm tra ngoại quan công nghiệp (Automated Surface Inspection).
- **Mục tiêu mô hình:**
  1. Huấn luyện Defect-GAN trên cặp miền ảnh không ghép đôi (Unpaired Image-to-Image Translation) giữa **F971 OK (Clean)** và **F971 NG (Scratch)**.
  2. Generator học cách sinh lỗi Scratch chân thực, có hình thái, độ sắc nét và màu sắc tự nhiên trên nền sản phẩm F971 OK.
  3. **Bảo toàn nền tuyệt đối (Background Preservation):** Giữ nguyên cấu trúc kim loại, lớp phủ (coating), lỗ, mép, và chữ in bên ngoài vùng lỗi thông qua cơ chế ghép lớp Layer-Wise Composition.
  4. Hỗ trợ dịch ngược phục hồi (Restoration): `F971 NG -> F971 OK`.
  5. Thử nghiệm sinh lỗi Scratch trên sản phẩm mã khác (**S942 OK**) bằng sliding-window patch inference.
- **Ràng buộc:** Toàn bộ dữ liệu đầu vào và ảnh sinh ra đều là ảnh màu **RGB 3 kênh**. Tuyệt đối không dùng binary mask PNG làm ảnh đích.

---

## 2. Dataset Baseline Đã Được Phê Duyệt (114 Patches)

Bộ dữ liệu baseline sử dụng cấu hình chuẩn: **1024 × 1024 native PNG RGB lossless, không resize**.
Được tổng hợp và kiểm định nghiêm ngặt từ báo cáo `reports/final_patch_review_v2/final_patch_review_report.txt`:

### Bảng phân bố Split chính xác:
| Split | F971 NG (Defect) | F971 OK (Clean) | Tổng Patch | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 31 | 41 | **72** | Sử dụng 39 patch OK thay thế không viền đen; 0 data leakage |
| **Validation** | 8 | 9 | **17** | Phân chia độc lập theo ảnh nguồn (source image) |
| **Test** | 8 | 17 | **25** | Dành riêng cho đánh giá cuối cùng |
| **TỔNG** | **47** | **67** | **114** | **100% RGB 3 kênh, 1024×1024** |

### Các loại trừ bắt buộc tuân thủ:
- **HOLD Patches:** P25, P29, P43 bị loại bỏ khỏi baseline do hình thái không rõ ràng hoặc chạm biên khung hình.
- **EDGE DROPPED Patches:** 16 patch có vết xước lân cận bị cắt cụt bởi biên crop 1024 (P5, P6, P7, P8, P9, P10, P11, P13, P16, P17, P18, P20, P30, P46, P47, P48) bị loại bỏ.
- **Highlighted NG Crop:** P45 (2026-09-10-16-10-51_crop001_x3482_y777.png) bị loại bỏ do tỷ lệ viền đen cao (>33%).
- **Clean Replacement Crops:** 39 crop OK ban đầu có viền đen đã được thay thế bằng các tọa độ mới trên cùng ảnh nguồn, đạt 0.000% viền đen.
- **Data Leakage Check:** 92 ảnh nguồn duy nhất; không có ảnh nguồn nào xuất hiện đồng thời ở nhiều hơn 1 split.

Manifests chính thức được lưu tại:
- `training/manifests/train.csv` (72 dòng)
- `training/manifests/val.csv` (17 dòng)
- `training/manifests/test.csv` (25 dòng)

---

## 3. Kiến Trúc Defect-GAN & Đối Chiếu Paper

Kiến trúc tuân thủ thiết kế trong bài báo gốc:
> **"Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"**  
> *Yuhang Shen, et al. (WACV 2021, arXiv: 2103.15158)*

### Bảng đối chiếu thành phần:
| Thành phần | Thiết kế trong Paper Defect-GAN | Triển khai trong Source Code (`src/`) |
| :--- | :--- | :--- |
| **Generator Architecture** | Encoder-Bottleneck-Decoder với Stride-8 downsampling | `DefectGANGenerator` (`src/models/generator.py`) |
| **Layer-Wise Composition** | Sinh riêng Foreground $F$ (RGB) và Spatial Mask $M \in [0, 1]$. Ảnh ra: $\hat{I} = (1 - M) \odot I_{bg} + M \odot F$ | Hoàn toàn đồng nhất: `out = (1 - mask) * input + mask * foreground` |
| **SPADE Normalization** | Điều chế đặc trưng decoder theo category vector hoặc spatial control map | `SPADENorm` (`src/models/generator.py`) với `num_classes=1` |
| **Adaptive Noise Injection** | Nhiễu Gaussian được bơm vào sau mỗi conv block với trọng số học được per-channel | `AdaptiveNoiseInjection` (`src/models/generator.py`) |
| **Dual Generators** | $G_{n \to d}$ (Defacement) và $G_{d \to n}$ (Restoration) | `DefectGAN.G_n2d` và `DefectGAN.G_d2n` riêng trọng số |
| **Discriminator** | Multi-Scale PatchGAN với nhánh phân loại phụ (Auxiliary Classifier) | `MultiscaleDiscriminator` (`src/models/discriminator.py`) 2 scales |
| **Adversarial Loss** | LSGAN (Least Squares GAN Loss / MSE) | `LSGANLoss` (`src/losses/gan_losses.py`) |
| **Classification Loss** | Binary Cross Entropy trên nhánh phụ của Discriminator | `ClassificationLoss` (`src/losses/gan_losses.py`) |
| **Cycle Consistency Loss** | L1 Loss giữa ảnh gốc và ảnh sau chu trình $n \to \hat{d} \to \hat{\hat{n}}$ | `CycleConsistencyLoss` (`src/losses/gan_losses.py`, $\lambda=10.0$) |
| **Identity / Rec Loss** | L1 Loss giữ nguyên ảnh khi đã thuộc domain đích | `IdentityLoss` (`src/losses/gan_losses.py`, $\lambda=10.0$) |
| **Mask Sparsity Loss** | Regularization L1 norm ép mask $M$ thưa, bảo vệ vùng nền không lỗi | `MaskSparsityLoss` (`src/losses/gan_losses.py`, $\lambda=0.1$) |

---

## 4. Cấu Trúc Mã Nguồn

```text
GAN_Project/
│
├── F971/                           # Ảnh nguồn F971 OK (137) và F971 NG (49)
├── S942/                           # Ảnh nguồn S942 OK (34)
├── experiments/
│   └── patches_1024x1024/          # Patches 1024x1024 gốc (train/val/test)
├── reports/                        # Báo cáo audit và candidate replacements v2
│
├── training/
│   └── manifests/
│       ├── train.csv               # 72 approved patches
│       ├── val.csv                 # 17 approved patches
│       └── test.csv                # 25 approved patches
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Seeding, image save/load, valid tensor checks
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── f971_dataset.py         # Unpaired RGB DataLoader
│   │   └── transforms.py           # Pure PyTorch/PIL RGB transforms
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py            # DefectGANGenerator (SPADE + Composition)
│   │   ├── discriminator.py        # MultiscaleDiscriminator (PatchGAN + AuxCls)
│   │   └── defect_gan.py           # DefectGAN composite wrapper
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── gan_losses.py           # LSGAN, Cls, Cycle, Identity, Mask losses
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py              # DefectGANTrainer (AMP, Accum, Checkpointing)
│   │   └── evaluator.py            # DefectGANEvaluator (Validation & RGB sample preview)
│   │
│   ├── train.py                    # Training CLI script
│   └── inference.py                # Inference & sliding-window blending script
│
├── configs/
│   ├── defectgan_1024.yaml         # Cấu hình huấn luyện 1024 baseline cho GPU
│   └── smoke_test.yaml             # Cấu hình kiểm thử CPU nhanh
│
├── tests/
│   ├── test_dataset.py             # Unit test dataset & manifest counts
│   ├── test_generator.py           # Unit test generator & composition
│   ├── test_discriminator.py       # Unit test multi-scale discriminator
│   ├── test_losses.py              # Unit test loss functions & gradients
│   └── test_pipeline.py            # Integration test trainer & inference
│
├── docker/
│   └── entrypoint.sh               # Entrypoint script cho container
│
├── scripts/
│   ├── validate_dataset.py         # Kiểm định dataset & sinh manifests
│   ├── check_environment.py        # Kiểm tra phần cứng, GPU, VRAM, CUDA
│   └── run_smoke_test.py           # Chạy toàn bộ 5 stage smoke test trên CPU
│
├── Dockerfile                      # PyTorch 2.1.2 CUDA 12.1 Linux container
├── docker-compose.yml              # GPU service & CPU smoke-test service
├── .dockerignore                   # Loại trừ datasets và artifacts khỏi image
├── .env.example                    # File mẫu thiết lập biến môi trường
├── requirements.txt                # Pinned dependencies
├── README.md                       # Tài liệu tổng quan
└── TRAINING_GUIDE.md               # Sổ tay hướng dẫn triển khai máy GPU công ty
```

---

## 5. Hướng Dẫn Phát Triển & Kiểm Thử Trên Windows

Máy phát triển cục bộ sử dụng CPU / Intel Iris Xe. **Không chạy full training trên máy cá nhân.**

### 5.1. Cài đặt môi trường Python
```powershell
# Sử dụng Python 3.11 hoặc 3.12
python -m pip install -r requirements.txt
```

### 5.2. Chạy kiểm tra môi trường
```powershell
python scripts/check_environment.py
```

### 5.3. Chạy kiểm định Dataset & Manifests
```powershell
python scripts/validate_dataset.py
```

### 5.4. Chạy toàn bộ PyTest Suite (17 Unit Tests)
```powershell
python -m pytest tests -v
```

### 5.5. Chạy CPU Smoke Test toàn diện
Kiểm tra forward, backward, loss, optimizer step, lưu checkpoint và chạy inference:
```powershell
python scripts/run_smoke_test.py
```
Kết quả kỳ vọng: **STATUS: READY_FOR_GPU_SMOKE_TEST**.

---

## 6. Đóng Gói & Chạy Docker

Dự án tuân theo nguyên tắc: **Phát triển trên Windows -> Đóng gói Docker image -> Chuyển sang máy GPU NVIDIA Linux**.

### 6.1. Build Docker Image
```bash
docker build -t defectgan:latest .
```

### 6.2. Kiểm tra container trên CPU (Smoke test)
```bash
docker run --rm \
  -v "D:/RTC/RD/GAN_Project:/workspace/data:ro" \
  -v "D:/RTC/RD/GAN_Project/outputs:/workspace/outputs" \
  defectgan:latest smoke-test
```

### 6.3. Khởi chạy GPU Training qua Docker Compose (Trên máy GPU công ty)
Xem chi tiết hướng dẫn đầy đủ trong [TRAINING_GUIDE.md](file:///D:/RTC/RD/GAN_Project/TRAINING_GUIDE.md).
