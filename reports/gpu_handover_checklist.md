# GPU handover checklist

- [ ] Xác nhận mã, config, log và image đúng phiên bản; đọc `verification_report.md` để biết trạng thái kiểm chứng hiện tại.
- [ ] Mount dataset đầy đủ tại `/workspace/data:ro` và output tại `/workspace/outputs` writable; chạy `validate-data` và xác nhận 114 patch, split 72/17/25, 39 replacements, không leakage/patch loại.
- [ ] Thu thập `nvidia-smi` và `check-env` với output được mount: GPU model/count, total/free VRAM, driver, torch/CUDA, FP16/BF16, OS và Docker. Đây là số liệu thực đo, không phải dự đoán khả năng chạy.
- [ ] Chạy container pytest và CPU smoke; ghi command, exit code và log.
- [ ] Chạy `gpu-smoke` như hướng dẫn; xác minh 1024 RGB batch, G/critic forward, các thành phần loss, backward, weight update G/D và loss/gradient hữu hạn.
- [ ] Ghi peak allocated/reserved VRAM, checkpoint và trạng thái resume, RGB inference và kiểm tra output ban đầu. Nếu một stage lỗi, không đánh dấu các stage phụ thuộc PASS.
- [ ] Chỉ sau smoke thành công, dựa trên số đo để xem xét batch, accumulation, AMP, workers và lịch training; không tự thay đổi kiến trúc hoặc dataset baseline để xử lý OOM.
- [ ] Lập kế hoạch đánh giá overfitting, background preservation, realism và domain shift S942; không dùng test set chọn model.

GPU smoke không chứng minh mô hình đã train thành công, Scratch chân thực hay khả năng chuyển giao S942. Không chạy full training như bước mặc định.
