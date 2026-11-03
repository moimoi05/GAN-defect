# Verification report

Date: 2026-11-03 (Asia/Bangkok).
Branch: `review/defectgan-gpu-smoke`
Base reviewed commit: `5f21e9a39be12d89d5518ce16e18ca26f17b41d8` (5f21e9a: "fix: Defect-GAN paper alignment and GPU readiness").
Working tree status: Targeted pre-GPU-smoke fixes applied to 7 files in working tree (not committed/pushed per user instruction). Tests below provide mechanical software pipeline validation, strictly distinguishing CPU logic verification from hardware GPU execution.

## Results

| Test / objective | Conditions / command | Result and evidence |
|---|---|---|
| Host full regression | Windows, Python 3.11.9, torch 2.9.1+cpu; `py -3.11 -m pytest -v` | **PASS**, exit 0, 110 passed, 1 skipped (CUDA hardware test skipped as expected on CPU). All 7 new regression tests passed. |
| Container build | Docker Desktop Linux x86_64, `docker build -t defectgan:review-smoke .` | **PASS**, exit 0. Pinned official base `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`. |
| Manifest consistency & Dataset validator | Docker container, read-only mount `/workspace/data`; `validate-data` | **PASS**, exit 0. Manifest SHA-256 consistency verified between image and DATA_DIR. 114 RGB PNG 1024, 39 replacements, 0 excluded patches, 92 unique sources, 0 cross-split leakage. |
| Container pytest | Docker container `defectgan:review-smoke pytest -v` | **PASS**, exit 0, 110 passed, 1 skipped in 51.99s. All data contract, component, pipeline, and trainer regression tests green. |
| Container CPU bounded smoke | Docker container `defectgan:review-smoke smoke-test` | **PASS**, exit 0, all 5 stages passed: Stage 1 Dataset PASS, Stage 2 Components PASS, Stage 3 Trainer Dry-Run PASS (D=5, G=1 updates), Stage 4 Safe Resume PASS (advanced to D=10, G=2 updates), Stage 5 Inference PASS (lossless RGB PNG). |
| Real GPU smoke / CUDA AMP / Peak VRAM | Local machine lacks NVIDIA GPU; company GPU environment pending | **NOT_RUN / PENDING GPU HANDOVER**. No claim of GPU PASS or VRAM confirmation. Hardware execution must be verified during GPU smoke step. |
| Full GAN training & quality transfer | Explicitly outside targeted fix scope | **NOT_RUN**. No full training run performed. |

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
docker build --progress plain -t defectgan:codex-retest .
docker run --rm --mount 'type=bind,src=D:\RTC\RD\GAN_Project,dst=/workspace/data,readonly' --mount 'type=bind,src=D:\RTC\RD\GAN_Project\outputs,dst=/workspace/outputs' -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 defectgan:codex-retest pytest -q
docker run --rm --mount 'type=bind,src=D:\RTC\RD\GAN_Project,dst=/workspace/data,readonly' --mount 'type=bind,src=D:\RTC\RD\GAN_Project\outputs,dst=/workspace/outputs' -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 defectgan:codex-retest smoke-test
```

Validation used the same mounts and `defectgan:codex-final validate-data` before the trainer-only autocast fix. Current source and retest image hashes were compared for training, model, loss, inference and smoke files. The initial `docker_build.log` records a separate simultaneous export error; the independently named subsequent builds succeeded as recorded above.
