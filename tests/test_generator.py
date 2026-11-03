"""Unit tests for Defect-GAN Generator architecture and sub-modules."""

import pytest
import torch

from src.models.generator import (
    DefectGANGenerator,
    AdaptiveNoiseInjection,
    SPADENorm,
    ResBlock,
)


def test_adaptive_noise_injection():
    """Verify AdaptiveNoiseInjection adds noise scaled by learnable weight."""
    layer = AdaptiveNoiseInjection(num_channels=32)
    x = torch.zeros(2, 32, 16, 16)

    # In eval mode, no noise
    layer.eval()
    assert torch.equal(layer(x), x)

    # In training mode, initially weight is 0 -> output is x
    layer.train()
    assert torch.equal(layer(x), x)

    # Once weight is non-zero, noise is added
    layer.weight.data.fill_(1.0)
    out = layer(x)
    assert not torch.equal(out, x)
    assert out.shape == x.shape


def test_spade_norm():
    """Verify SPADENorm normalises features conditioned on attribute map."""
    norm = SPADENorm(norm_channels=64, attr_channels=1)
    x = torch.randn(2, 64, 16, 16)
    attr = torch.ones(2, 1, 8, 8)  # Will be resized internally

    out = norm(x, attr)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_generator_forward_and_composition():
    """Verify Generator output shape, range, and layer-wise composition formula."""
    generator = DefectGANGenerator(
        in_channels=3,
        base_channels=16,
        num_classes=1,
        num_res_blocks=2,
    )
    generator.eval()

    b, c, h, w = 2, 3, 64, 64
    x = torch.randn(b, c, h, w).clamp(-1.0, 1.0)
    category = torch.ones(b, 1)

    res = generator(x, category)

    assert "output" in res
    assert "foreground" in res
    assert "mask" in res

    out = res["output"]
    fg = res["foreground"]
    mask = res["mask"]

    assert out.shape == (b, 3, h, w)
    assert fg.shape == (b, 3, h, w)
    assert mask.shape == (b, 1, h, w)

    # Output range checks
    assert -1.05 <= out.min().item() and out.max().item() <= 1.05, "Output out of Tanh range"
    assert -1.05 <= fg.min().item() and fg.max().item() <= 1.05, "Foreground out of Tanh range"
    assert 0.0 <= mask.min().item() and mask.max().item() <= 1.0, "Mask out of Sigmoid range"

    # Layer-wise composition formula verification: out = (1 - mask) * x + mask * fg
    expected_out = (1.0 - mask) * x + mask * fg
    assert torch.allclose(out, expected_out, atol=1e-5), "Composition formula mismatch!"


def test_generator_backward_pass():
    """Verify gradients flow through generator back to inputs and parameters."""
    generator = DefectGANGenerator(
        in_channels=3,
        base_channels=16,
        num_classes=1,
        num_res_blocks=2,
    )
    generator.train()

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    res = generator(x)
    loss = res["output"].mean() + res["mask"].mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Check parameter gradients
    for name, p in generator.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for parameter: {name}"
            assert torch.isfinite(p.grad).all()
