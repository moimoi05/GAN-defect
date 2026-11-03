"""End-to-End CPU Smoke Test Runner for Defect-GAN Pipeline.

Validates that all pipeline stages run without error on CPU:
  Stage 1: Dataset & Manifest verification (114 approved patches).
  Stage 2: Model component unit checks (Generator, Discriminator, Losses).
  Stage 3: Trainer dry-run execution (forward, backward, loss, optimizer step).
  Stage 4: Checkpoint save & load verification.
  Stage 5: Inference verification (single image + patch blending).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(text: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def run_stage_1_dataset() -> bool:
    print_banner("STAGE 1: DATASET MANIFEST & INTEGRITY AUDIT")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/validate_dataset.py")]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return res.returncode == 0


def run_stage_2_components() -> bool:
    print_banner("STAGE 2: MODEL ARCHITECTURE & LOSS FORWARD/BACKWARD CHECK")
    from src.models.generator import DefectGANGenerator
    from src.models.discriminator import MultiscaleDiscriminator
    from src.models.defect_gan import DefectGAN
    from src.losses.gan_losses import DefectGANLoss

    device = torch.device("cpu")
    b, c, h, w = 1, 3, 128, 128
    dummy_normal = torch.randn(b, c, h, w, device=device).clamp(-1.0, 1.0)
    dummy_defect = torch.randn(b, c, h, w, device=device).clamp(-1.0, 1.0)
    dummy_cls = torch.ones(b, 1, device=device)

    print("Testing Generator forward & backward...")
    net_g = DefectGANGenerator(in_channels=3, base_channels=16, num_classes=1, num_res_blocks=2).to(device)
    g_out = net_g(dummy_normal, dummy_cls)
    assert g_out["output"].shape == (b, c, h, w), f"Bad G output shape: {g_out['output'].shape}"
    assert g_out["mask"].shape == (b, 1, h, w), f"Bad G mask shape: {g_out['mask'].shape}"
    assert g_out["foreground"].shape == (b, c, h, w), f"Bad G fg shape: {g_out['foreground'].shape}"
    assert -1.01 <= g_out["output"].min().item() and g_out["output"].max().item() <= 1.01, "G output out of [-1, 1]"
    assert 0.0 <= g_out["mask"].min().item() and g_out["mask"].max().item() <= 1.0, "G mask out of [0, 1]"
    loss_g_test = g_out["output"].sum()
    loss_g_test.backward()
    print("  Generator forward & backward: PASS")

    print("Testing Discriminator forward & backward...")
    net_d = MultiscaleDiscriminator(in_channels=3, base_channels=16, num_classes=1, num_scales=2).to(device)
    d_out = net_d(dummy_defect)
    assert len(d_out) == 2, f"Expected 2 scales, got {len(d_out)}"
    assert d_out[0][0].ndim == 4, f"Scale 0 adv should be 4D: {d_out[0][0].shape}"
    assert d_out[0][1].shape == (b, 1), f"Scale 0 cls logits shape mismatch: {d_out[0][1].shape}"
    loss_d_test = d_out[0][0].sum() + d_out[0][1].sum()
    loss_d_test.backward()
    print("  Discriminator forward & backward: PASS")

    print("Testing DefectGAN composite wrapper & DefectGANLoss...")
    model = DefectGAN(in_channels=3, base_channels_g=16, base_channels_d=16, num_classes=1, num_res_blocks=2, num_disc_scales=1).to(device)
    loss_fn = DefectGANLoss()
    res = model(dummy_normal, dummy_defect, dummy_cls)
    d_fake = model.discriminate_defect(res["fake_defect"]["output"])
    d_norm_fake = model.discriminate_normal(res["fake_normal"]["output"])
    g_loss, g_dict = loss_fn.compute_generator_loss(
        normal_img=dummy_normal,
        defect_img=dummy_defect,
        fake_defect=res["fake_defect"],
        fake_normal=res["fake_normal"],
        recon_normal=res["recon_normal"],
        recon_defect=res["recon_defect"],
        id_defect=None,
        id_normal=None,
        d_defect_fake=d_fake,
        d_normal_fake=d_norm_fake,
        target_cls_defect=dummy_cls,
        target_cls_normal=torch.zeros_like(dummy_cls),
    )
    assert torch.isfinite(g_loss), "G loss is not finite!"
    g_loss.backward()
    print(f"  Composite forward, loss computation ({g_loss.item():.4f}), & backward: PASS")
    return True


def run_stage_3_trainer_dryrun() -> bool:
    print_banner("STAGE 3: TRAINER DRY-RUN EXECUTION (CPU)")
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        "configs/smoke_test.yaml",
        "--dry-run",
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return res.returncode == 0


def run_stage_4_checkpoint_verify() -> bool:
    print_banner("STAGE 4: CHECKPOINT SAVE & LOAD VERIFICATION")
    ckpt_dir = PROJECT_ROOT / "outputs/smoke_test/smoke_test_cpu/checkpoints"
    latest_ckpt = ckpt_dir / "latest.pth"
    if not latest_ckpt.exists():
        print(f"FAILED: latest.pth was not created at {latest_ckpt}")
        return False

    print(f"Found checkpoint: {latest_ckpt} ({latest_ckpt.stat().st_size:,} bytes)")
    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    required_keys = ["epoch", "global_step", "model_state", "opt_g_state", "opt_d_state", "config"]
    for k in required_keys:
        assert k in checkpoint, f"Missing key in checkpoint: {k}"

    print(f"Checkpoint structure verified. Saved at epoch {checkpoint['epoch']}, step {checkpoint['global_step']}.")
    return True


def run_stage_5_inference() -> bool:
    print_banner("STAGE 5: INFERENCE PIPELINE VERIFICATION")
    latest_ckpt = PROJECT_ROOT / "outputs/smoke_test/smoke_test_cpu/checkpoints/latest.pth"
    out_dir = PROJECT_ROOT / "outputs/smoke_test/inference_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use first train clean image from manifest as sample test input
    import csv
    with open(PROJECT_ROOT / "training/manifests/train.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sample_rel_path = next(r["patch_path"] for r in reader if r["domain"] == "clean")

    sample_input = PROJECT_ROOT / sample_rel_path
    sample_out = out_dir / "test_synth_defect.png"

    cmd = [
        sys.executable,
        "-m",
        "src.inference",
        "--checkpoint",
        str(latest_ckpt),
        "--input",
        str(sample_input),
        "--output",
        str(sample_out),
        "--mode",
        "defacement",
        "--patch-size",
        "128",
        "--stride",
        "64",
        "--save-mask",
        "--device",
        "cpu",
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        return False

    if not sample_out.exists():
        print(f"FAILED: Inference output {sample_out} was not generated!")
        return False

    from PIL import Image
    with Image.open(sample_out) as img:
        print(f"Inference output generated: {sample_out.name}, size={img.size}, mode={img.mode}")
        assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"

    return True


def main() -> None:
    results = {}
    results["Stage 1 (Dataset & Manifests)"] = run_stage_1_dataset()
    results["Stage 2 (Model Components)"] = run_stage_2_components()
    results["Stage 3 (Trainer Dry-run)"] = run_stage_3_trainer_dryrun()
    results["Stage 4 (Checkpoint Save/Load)"] = run_stage_4_checkpoint_verify()
    results["Stage 5 (Inference Pipeline)"] = run_stage_5_inference()

    print_banner("FINAL CPU SMOKE TEST SUMMARY")
    all_passed = True
    for stage, passed in results.items():
        status = "PASSED [OK]" if passed else "FAILED [X]"
        print(f"  {stage:.<45} {status}")
        if not passed:
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("  STATUS: READY_FOR_GPU_SMOKE_TEST")
        print("  All local CPU development smoke tests passed completely.")
    else:
        print("  STATUS: NOT_READY (Fix indicated errors above)")
    print("=" * 65)


if __name__ == "__main__":
    main()
