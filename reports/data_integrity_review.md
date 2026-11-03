# Dataset integrity and container filesystem review

Reviewed against local `adccccf898bf1a14fc4235cf4df3f2a76f43b34b`, initially clean working tree. No approved manifest, source image, source COCO JSON, or prior patch-review report was edited. Dataset validation is read-only by default.

## Measured dataset result

`py -3.11 scripts/validate_dataset.py` completed with exit 0. Evidence: `audit_logs/dataset_validation.log`.

| Split | NG Scratch | OK | Total |
|---|---:|---:|---:|
| Train | 31 | 41 | 72 |
| Validation | 8 | 9 | 17 |
| Test | 8 | 17 | 25 |
| Total | 47 | 67 | 114 |

All 114 image files exist and were fully decoded as PNG RGB at native 1024×1024. Existing manifest rows exactly match the approved clean review and approved defect inventory, including source, split, filename and relative path. All 39 `REPLACE_FROM_SOURCE_RGB` clean replacements remain selected. The three HOLD patches, sixteen edge-clipped patches and P45 are absent. Oversize proposal images are absent from the exact approved identity set. There are 92 distinct source names and zero cross-split source collisions.

Before tests, SHA256 was captured for 729 files under F971, S942, experiments, both prior final-review directories, training/manifests, and existing top-level report files. Full before inventory: `../outputs/audit/protected_sha256_before.json`. Rehash result: `audit_logs/protected_integrity.json` (zero changed or missing). The full inventory includes original source images and JSON, not only the selected training patches. Parent integration verification should repeat this same comparison after all smoke stages.

## Changes and contract

- `scripts/validate_dataset.py`: default validation reads existing manifests and compares against approved review records; it never regenerates approved manifests. Failures raise and return nonzero through the CLI. `DATA_DIR` or `--data-root` selects data independently of code root. `--manifest-dir` permits immutable manifests inside the image while resolving images under the mounted dataset.
- `--generate-manifests --output-dir <new-empty-directory>` is explicit and refuses protected dataset directories and nonempty outputs. It does not overwrite approved manifests.
- `src/datasets/f971_dataset.py`: strict root-relative portable manifest paths (Windows separators normalized), absolute/drive-qualified paths and escapes rejected, no fallback to code-root data, split validation and rejection of non-RGB/non-PNG inputs. File context managers close image handles. Evaluation remains deterministic; training continues unpaired random defect sampling. Debug resizing affects tensors only and is explicit via image_size, never stored patches.
- Docker code `/workspace`, dataset `/workspace/data:ro`, output `/workspace/outputs` writable. Both CPU and GPU services preserve this contract. Compose services require explicit profiles; `docker compose up` cannot start full GAN training by default.
- Entry point uses `set -euo pipefail`, safely handles zero arguments, forwards flags, and supports `validate-data` and bounded `gpu-smoke` commands.

## Dependency decision

Retain the existing official PyTorch 2.1.2 / CUDA 12.1 / cuDNN8 runtime, pinned to the registry-resolved immutable image digest. This limits changes while the trainer uses APIs supported by 2.1.2. Torch 2.1.2 supports host CPython through 3.11; use Python 3.11 rather than the machine's default Python 3.12. NumPy 1.26.4 avoids NumPy2 ABI incompatibility with this Torch generation. Pillow10.4, OpenCV-headless4.10.0.84, PyYAML6.0.2 and test tools are explicitly pinned. These are compatibility choices, not a claim to use newest releases or a security certification. OS apt repositories and preinstalled image Python dependencies are not separately locked; base digest and explicit direct pins provide the documented reproducibility boundary.

## Verification evidence

- RED: `audit_logs/data_red.log`, seven intended failures before production edits (path rejection, split rejection, RGB contract, read-only validation API and protected manifest generation).
- GREEN and coverage: `audit_logs/data_tests.log`, actual result at tail.
- Dataset validation: `audit_logs/dataset_validation.log`, exit0.
- Docker Compose syntax: `docker compose config --quiet`, exit0.
- Docker build: `audit_logs/docker_build.log`; status must be taken from actual final build evidence and parent verification report, never inferred from local tests.

Container execution needs the complete approved data tree (including `reports/final_patch_review_v2/candidate_patches_rgb`), not only F971 source images. Example:

```bash
docker run --rm -v "<ABS_DATA_ROOT>:/workspace/data:ro" -v "<ABS_OUTPUT_ROOT>:/workspace/outputs" defectgan:audit validate-data
```

Small tensor tests validate portable path, sampling, normalization and error contracts; they do not establish GPU memory fitness, scratch realism, or transfer to S942.

Primary dependency reference: [PyTorch previous-version installation matrix](https://pytorch.org/get-started/previous-versions/) lists PyTorch2.1.2 CUDA12.1 distribution. The actual container version/import evidence is reported separately after execution.

## Completed container verification

Final tested image: `defectgan:audit-final-20261103` (manifest digest `sha256:5b815fee73cce4a039a2101d73f6aa73dc1fa262296a1d584b42649f09745dd6`).

| Check | Actual outcome | Evidence |
|---|---|---|
| Final Docker build | PASS, exit0 | `audit_logs/docker_build_final.log` |
| Runtime versions/imports | PASS, exit0; CPython3.10.13, Torch2.1.2, CUDA runtime12.1, CUDA unavailable | `audit_logs/docker_versions.log` |
| Read-only approved dataset validation | PASS, exit0;114 patches,39 replacements,92 sources | `audit_logs/docker_dataset_validation.log` |
| Data mount read-only/output mount writable | PASS, exit0; OS mount flag checked, output probe written | `audit_logs/docker_mount_contract.log` |
| Full Docker pytest | PASS,103 tests,40.19s,exit0 | `audit_logs/docker_pytest.log`, `audit_logs/docker_pytest.xml` |
| Docker CPU smoke | PASS,all5stages,exit0 | `audit_logs/docker_cpu_smoke.log` |
| Tested image source equals host | PASS;37 source/config/script/test/manifest files,zero differences | `audit_logs/docker_source_comparison.json` |
| Protected original files after container tests | PASS;729 files,zero changes | `audit_logs/protected_integrity.json` |
| GPU smoke | NOT_RUN;CUDA unavailable locally | `audit_logs/docker_versions.log` |

Smoke checkpoint evidence at `../outputs/docker_cpu_smoke/_smoke/smoke_test_cpu/dry_run_evidence.json` reports G and D weights changed,2 generator updates,10 critic updates after resume,finite losses,safe epoch-boundary checkpoint schema2 and smoke-only status. RGB inference output preserves the synthetic input dimensions (41×35). This is a mechanics smoke; no trained image-quality claim follows.

Initial Docker build failed at the image exporter due to an existing-tag collision; retry with a distinct tag succeeded. Initial container tests also exposed a real Torch2.1.2 CPU autocast incompatibility (17 test failures), despite local Torch2.9 tests. Trainer was corrected to use a null context when AMP is disabled, then the complete container tests/smoke were rerun successfully. Initial failure evidence remains in `audit_logs/docker_pytest_initial.log`, `audit_logs/docker_pytest_initial.xml`, and `audit_logs/docker_cpu_smoke_initial.log`.
