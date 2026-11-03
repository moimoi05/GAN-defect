"""Unit tests for Defect-GAN Discriminator architecture."""

import pytest
import torch

from src.models.discriminator import PatchGANDiscriminator, MultiscaleDiscriminator


def test_single_scale_discriminator():
    """Verify single PatchGAN outputs adversarial patch map and class logits."""
    disc = PatchGANDiscriminator(in_channels=3, base_channels=16, num_classes=1)
    x = torch.randn(2, 3, 64, 64)

    adv, cls_logits = disc(x)

    assert adv.ndim == 4
    assert adv.shape[0] == 2
    assert adv.shape[1] == 1  # 1 channel adv score
    assert cls_logits.shape == (2, 1)  # Auxiliary class logits

    assert torch.isfinite(adv).all()
    assert torch.isfinite(cls_logits).all()


def test_multiscale_discriminator():
    """Verify MultiscaleDiscriminator produces outputs across scales."""
    m_disc = MultiscaleDiscriminator(in_channels=3, base_channels=16, num_classes=1, num_scales=2)
    x = torch.randn(2, 3, 64, 64)

    results = m_disc(x)

    assert len(results) == 2, f"Expected 2 scale outputs, got {len(results)}"
    scale0_adv, scale0_cls = results[0]
    scale1_adv, scale1_cls = results[1]

    assert scale0_adv.shape[0] == 2
    assert scale1_adv.shape[0] == 2
    # Scale 1 spatial size should be smaller than scale 0
    assert scale1_adv.shape[2] < scale0_adv.shape[2]
    assert scale1_adv.shape[3] < scale0_adv.shape[3]


def test_discriminator_backward():
    """Verify gradient computation through discriminator."""
    m_disc = MultiscaleDiscriminator(in_channels=3, base_channels=16, num_classes=1, num_scales=2)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)

    results = m_disc(x)
    loss = sum(adv.mean() + cls.mean() for adv, cls in results)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
