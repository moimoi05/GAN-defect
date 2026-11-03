"""Unit tests for Defect-GAN loss functions."""

import pytest
import torch

from src.losses.gan_losses import (
    LSGANLoss,
    ClassificationLoss,
    CycleConsistencyLoss,
    IdentityLoss,
    MaskSparsityLoss,
    DefectGANLoss,
)


def test_lsgan_loss():
    """Verify LSGAN loss computation for generator and discriminator."""
    lsgan = LSGANLoss()

    real_pred = [torch.ones(2, 1, 8, 8)]
    fake_pred = [torch.zeros(2, 1, 8, 8)]

    # Perfect predictions -> 0 loss
    d_loss_perfect = lsgan.discriminator_loss(real_pred, fake_pred)
    assert d_loss_perfect.item() == pytest.approx(0.0, abs=1e-5)

    g_loss = lsgan.generator_loss(fake_pred)
    assert g_loss.item() > 0.0


def test_mask_sparsity_loss():
    """Verify mask sparsity loss penalizes non-zero alpha masks."""
    loss_fn = MaskSparsityLoss()

    zero_mask = torch.zeros(2, 1, 32, 32)
    assert loss_fn(zero_mask).item() == 0.0

    dense_mask = torch.ones(2, 1, 32, 32)
    assert loss_fn(dense_mask).item() == 1.0


def test_composite_defect_gan_loss():
    """Verify DefectGANLoss computes all components and enables backprop."""
    criterion = DefectGANLoss(
        lambda_adv=1.0,
        lambda_cls=1.0,
        lambda_cyc=10.0,
        lambda_rec=10.0,
        lambda_mask=0.1,
    )

    b, c, h, w = 2, 3, 32, 32
    normal = torch.randn(b, c, h, w, requires_grad=True)
    defect = torch.randn(b, c, h, w, requires_grad=True)

    raw_mask_def = torch.randn(b, 1, h, w, requires_grad=True)
    raw_mask_norm = torch.randn(b, 1, h, w, requires_grad=True)

    fake_defect = {
        "output": torch.randn(b, c, h, w, requires_grad=True),
        "mask": torch.sigmoid(raw_mask_def),
        "foreground": torch.randn(b, c, h, w, requires_grad=True),
    }
    fake_normal = {
        "output": torch.randn(b, c, h, w, requires_grad=True),
        "mask": torch.sigmoid(raw_mask_norm),
        "foreground": torch.randn(b, c, h, w, requires_grad=True),
    }
    recon_normal = {"output": torch.randn(b, c, h, w, requires_grad=True)}
    recon_defect = {"output": torch.randn(b, c, h, w, requires_grad=True)}

    d_fake_defect = [(torch.randn(b, 1, 4, 4), torch.randn(b, 1))]
    d_fake_normal = [(torch.randn(b, 1, 4, 4), torch.randn(b, 1))]

    target_defect = torch.ones(b, 1)
    target_normal = torch.zeros(b, 1)

    g_loss, g_dict = criterion.compute_generator_loss(
        normal_img=normal,
        defect_img=defect,
        fake_defect=fake_defect,
        fake_normal=fake_normal,
        recon_normal=recon_normal,
        recon_defect=recon_defect,
        id_defect=fake_defect,
        id_normal=fake_normal,
        d_defect_fake=d_fake_defect,
        d_normal_fake=d_fake_normal,
        target_cls_defect=target_defect,
        target_cls_normal=target_normal,
    )

    assert torch.isfinite(g_loss)
    assert "g_loss" in g_dict
    assert "g_adv_defect" in g_dict
    assert "g_cyc_normal" in g_dict
    assert "g_mask_defect" in g_dict

    # Check backpropagation
    g_loss.backward()
    assert fake_defect["output"].grad is not None
    assert raw_mask_def.grad is not None
