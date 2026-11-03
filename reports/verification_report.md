# Verification report

Date: 2026-11-03 (Asia/Bangkok). HEAD `adccccf898bf1a14fc4235cf4df3f2a76f43b34b`. Working tree already contained substantial uncommitted changes at start of this session; see `git status --short`. Tests below are mechanical validation, not trained-model or image-quality evidence.

## Results

| Test / objective | Conditions / command | Result and evidence |
|---|---|---|
| Host full regression and coverage | Windows, Python 3.11.9, torch 2.9.1+cpu; `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 py -3.11 -m pytest -q --cov=src --cov-report=term --junitxml=outputs/audit/final_pytest.xml` (PowerShell environment variables) | **PASS**, exit 0, 103 passed, total statement coverage 83%; per-file limits and 40 deprecation warnings in `outputs/audit/final_pytest.log`. Named cases and durations in `outputs/audit/final_pytest.xml`. Coverage is not proof of correctness; `src/train.py` 27%, inference 74%. |
| Container build | Docker Desktop Linux x86_64, `docker build -t defectgan:codex-retest .` | **PASS**, exit 0; `outputs/docker_codex_retest_build.log`. Container Python 3.10.13, torch 2.1.2/CUDA runtime 12.1, NumPy 1.26.4, Pillow 10.4.0, PyYAML 6.0.2: `outputs/docker_versions.log` (same pinned base/dependencies). |
| Dataset validator | Docker image, complete dataset read-only mounted `/workspace/data`; `validate-data` | **PASS**, exit 0; `outputs/docker_validation.log`: 114 RGB PNG 1024, 39 replacements, no excluded patches, 92 distinct sources, no cross-split source leakage. Split train 31 NG/41 OK, validation 8/9, test 8/17. |
| Container pytest | Rebuilt image, dataset read-only; `pytest -q` | **PASS**, exit 0; 103 passed in 44.65s, `outputs/docker_pytest_retest.log`. Individual named cases are listed there. |
| Container CPU smoke | Rebuilt image, dataset read-only and writable output; `smoke-test` | **PASS**, exit 0; all five stages and per-stage exit codes in `outputs/docker_smoke_retest.log`. Reduced 32px model, G/D updates 1/5; safe smoke-only checkpoint resume to 2/10; finite losses, synthetic RGB output. This is not native 1024 performance. |
| Initial container attempts | Earlier image, same CPU smoke and pytest | **FAIL**, smoke and 17 tests: torch 2.1 CPU autocast rejected unsupported dtype even when disabled. Logs `outputs/docker_smoke.log`, `outputs/docker_pytest.log`. Fixed with `nullcontext`, regression added; rebuilt retest results above. |
| Dataset integrity | SHA-256 compare 729 paths to pre-existing `outputs/audit/protected_sha256_before.json` after tests | **PASS**, no changed/missing paths, `outputs/audit/current_integrity.json`. Comparison is relative to the recorded inventory, not proof of provenance before it was created. |
| GPU smoke / AMP / 1024 peak VRAM | NVIDIA GPU unavailable locally; company GPU unspecified | **NOT_RUN**. No CUDA optimizer, FP16/BF16 hardware execution, peak VRAM or real S942 image benchmark. |
| Full GAN training, quality/transfer validation | Explicitly outside this task | **NOT_RUN**. No claim of model training success, scratch realism or S942 transfer. |

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
