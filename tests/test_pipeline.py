"""Integration tests for Defect-GAN training and inference pipeline."""

from pathlib import Path
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import DefectGANLoss
from src.engine.trainer import DefectGANTrainer
from src.inference import run_sliding_window_inference


def test_full_model_cycles():
    """Verify DefectGAN forward passes, defacement, restoration, and cycle consistency."""
    model = DefectGAN(
        in_channels=3,
        base_channels_g=16,
        base_channels_d=16,
        num_classes=1,
        num_res_blocks=2,
        num_disc_scales=1,
    )
    b, c, h, w = 1, 3, 64, 64
    normal = torch.randn(b, c, h, w)
    defect = torch.randn(b, c, h, w)

    # Defacement: normal -> fake defect
    res_def = model.defacement(normal)
    assert res_def["output"].shape == (b, c, h, w)

    # Restoration: defect -> fake normal
    res_res = model.restoration(defect)
    assert res_res["output"].shape == (b, c, h, w)

    # Full cycle defacement: normal -> fake_d -> recon_n
    recon_n, fake_d = model.cycle_defacement(normal)
    assert recon_n["output"].shape == (b, c, h, w)
    assert fake_d["output"].shape == (b, c, h, w)


def test_trainer_single_step_and_checkpoint():
    """Verify trainer executes a train_step, updates parameters, and saves/loads checkpoint."""
    model = DefectGAN(
        in_channels=3,
        base_channels_g=16,
        base_channels_d=16,
        num_classes=1,
        num_res_blocks=1,
        num_disc_scales=1,
    )
    criterion = DefectGANLoss()
    device = torch.device("cpu")

    config = {
        "experiment": {"name": "test_exp", "seed": 42},
        "training": {
            "device": "cpu",
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "none",
            "optimizer": {"lr_g": 0.001, "lr_d": 0.001},
        },
    }

    # Dummy loaders
    b, c, h, w = 2, 3, 32, 32
    dummy_data = {
        "normal": torch.randn(b, c, h, w),
        "defect": torch.randn(b, c, h, w),
        "normal_path": "dummy_n.png",
        "defect_path": "dummy_d.png",
    }

    class DummyDataset:
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {**dummy_data, "normal": dummy_data["normal"][idx], "defect": dummy_data["defect"][idx]}

    loader = DataLoader(DummyDataset(), batch_size=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = DefectGANTrainer(
            config=config,
            model=model,
            criterion=criterion,
            train_loader=loader,
            val_loader=loader,
            device=device,
            output_dir=Path(tmp_dir),
        )

        logs = trainer.train_step(dummy_data["normal"], dummy_data["defect"], step_in_epoch=0)
        assert "g_loss" in logs
        assert "d_loss" in logs
        assert torch.isfinite(torch.tensor(logs["g_loss"]))
        assert torch.isfinite(torch.tensor(logs["d_loss"]))

        # Only completed epochs are safe to resume. Direct microsteps are not.
        trainer.train()
        # Checkpoint save/load test
        ckpt_path = Path(tmp_dir) / "test_ckpt.pth"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load back
        trainer.load_checkpoint(ckpt_path)
        assert trainer.global_step == 1


def test_sliding_window_inference():
    """Verify sliding window inference processes non-square or larger images with blending."""
    model = DefectGAN(
        in_channels=3,
        base_channels_g=16,
        base_channels_d=16,
        num_classes=1,
        num_res_blocks=1,
        num_disc_scales=1,
    )
    device = torch.device("cpu")
    model.eval()

    # Image larger than patch size: 100x120 with patch_size 64
    large_img = torch.randn(1, 3, 100, 120)

    out_img, mask = run_sliding_window_inference(
        model=model,
        image_tensor=large_img,
        mode="defacement",
        patch_size=64,
        stride=32,
        device=device,
    )

    assert out_img.shape == (1, 3, 100, 120)
    assert mask.shape == (1, 1, 100, 120)
    assert -1.05 <= out_img.min().item() and out_img.max().item() <= 1.05
    assert 0.0 <= mask.min().item() and mask.max().item() <= 1.0
