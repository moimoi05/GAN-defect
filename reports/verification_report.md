# Verification report

Date: 2026-11-03 (Asia/Bangkok).
Branch: `review/defectgan-gpu-smoke`
Base reviewed commit: `1cb78f8` ("test: add comprehensive regression suite for Defect-GAN trainer")
Working tree status: Targeted instrumentation fixes applied to 7 files in working tree (not committed/pushed per user instruction). Tests below provide mechanical software pipeline validation, strictly distinguishing CPU logic simulation from hardware GPU execution.

## Results

| Test / objective | Conditions / command | Result and evidence |
|---|---|---|
| Host full regression | Windows, Python 3.11.9, torch 2.9.1+cpu; `& "C:\Users\nnam05\AppData\Local\Programs\Python\Python311\python.exe" -m pytest -v` | **PASS**, exit 0, 113 passed, 1 skipped (CUDA hardware test skipped as expected on CPU). 0 failed. All regression tests passed. |
| Container build | Docker Desktop Linux x86_64, `docker build -t defectgan:review-smoke .` | **PASS**, exit 0. Pinned official base `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`. |
| Manifest consistency & Dataset validator | Docker container, read-only mount `/workspace/data`; `validate-data` | **PASS**, exit 0. Manifest SHA-256 consistency verified between image and DATA_DIR. 114 RGB PNG 1024, 39 replacements, 0 excluded patches, 92 unique sources, 0 cross-split leakage. |
| Container pytest | Docker container `defectgan:review-smoke pytest -v` | **PASS**, exit 0, 113 passed, 1 skipped in 87.61s. All data contract, component, pipeline, and trainer regression tests green. |
| Container CPU bounded smoke | Docker container `defectgan:review-smoke smoke-test` | **PASS**, exit 0, all 5 stages passed: Stage 1 Dataset PASS, Stage 2 Components PASS, Stage 3 Trainer Dry-Run PASS (D=5, G=1 updates), Stage 4 Safe Resume PASS (advanced to D=10, G=2 updates), Stage 5 Inference PASS (lossless RGB PNG). |
| Real GPU smoke / CUDA AMP / Peak VRAM | Local machine lacks NVIDIA GPU; company GPU environment pending | **NOT_RUN / PENDING GPU HANDOVER**. `test_real_cuda_amp_gradscaler_overflow` explicitly marked `NOT_RUN` / skipped until real CUDA hardware execution. Peak VRAM numbers pending GPU run. |
| Full GAN training & quality transfer | Explicitly outside targeted fix scope | **NOT_RUN**. No full training run performed. |

## Targeted Instrumentation Fixes Detail

1. **P0 — CUDA Peak VRAM Tracking (`src/engine/trainer.py`, `src/engine/evaluator.py`):**
   - In PyTorch, calling `torch.cuda.reset_peak_memory_stats(self.device)` resets peak counters. Previously, the evaluator reset at the start of validation was wiping out the peak memory recorded during training activations and backpropagation.
   - Fixed by capturing `train_peak_alloc` and `train_peak_res` immediately upon exiting the microbatch training loop, prior to validation.
   - Evaluator records validation-specific peak memory (`val_peak_cuda_allocated_bytes`, `val_peak_cuda_reserved_bytes`).
   - `trainer.py` calculates overall peak as `max(train_peak, val_peak)` and writes all separate metrics (`peak_cuda_train_*`, `peak_cuda_val_*`, `peak_cuda_overall_*`) plus legacy aliases into `dry_run_evidence.json`.
   - Regression tests `test_peak_cuda_vram_train_val_overall_tracking` and `test_evaluator_peak_cuda_vram_reporting` verify that training peak memory is preserved despite validation peak reset.

2. **P1 — CUDA FP16 Gate (`scripts/run_smoke_test.py`, `reports/gpu_handover_checklist.md`):**
   - Added `test_real_cuda_amp_gradscaler_overflow` as an explicit mandatory gate in `scripts/run_smoke_test.py --gpu-smoke` and in `reports/gpu_handover_checklist.md`.
   - On non-GPU environments (such as the local CPU machine), the hardware gate is marked `NOT_RUN` (skipped with `pytest.mark.skipif(not torch.cuda.is_available())`).
   - CPU mocks are strictly treated as logic simulation, not hardware proof of FP16 execution.

3. **P1 — Manifest Absolute Path (`src/datasets/f971_dataset.py`, `tests/test_data_contracts.py`):**
   - Previously, if an absolute path was passed for `manifest_path`, the consistency check between image workspace and `DATA_DIR` was bypassed because only a single candidate was retained.
   - Fixed `verify_manifest_consistency()` to identify counterparts across `DATA_DIR`, `project_root`, and relative subpaths even when an absolute path is provided.
   - Added `test_manifest_consistency_absolute_path_mismatch_fails_closed()` with two distinct temporary files containing differing contents, confirming fail-closed `ValueError` when an absolute path is used.

The default host Python 3.12 lacks torch, so use `py -3.11` for host commands. The torch 2.9 host warns that `torch.cuda.amp.GradScaler` is deprecated; this API was selected for the pinned 2.1.2 image. GPU hardware compatibility remains unverified. Validation critic loss omits GP; checkpoint selection uses reconstruction proxy and does not measure scratch realism. Safe smoke checkpoints are isolated from full-loader training. The project-specific 1024 configuration and original paper differences are documented in `paper_alignment_review.md`.

## Re-run on Linux

```bash
docker build -t defectgan:recheck .
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:recheck validate-data
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:recheck pytest -q
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:recheck smoke-test
```

Use the commands and gates in `gpu_handover_checklist.md` and `TRAINING_GUIDE.md` for the bounded GPU smoke. Do not launch full training as the next default step.

## Commands actually run in Docker Desktop (PowerShell)

```powershell
docker build -t defectgan:review-smoke .
docker run --rm -v "D:\RTC\RD\GAN_Project:/workspace/data:ro" defectgan:review-smoke validate-data
docker run --rm -v "D:\RTC\RD\GAN_Project:/workspace/data:ro" defectgan:review-smoke pytest -v
docker run --rm -v "D:\RTC\RD\GAN_Project:/workspace/data:ro" -v "D:\RTC\RD\GAN_Project\outputs:/workspace/outputs" defectgan:review-smoke smoke-test
```
