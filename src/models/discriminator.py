"""Multi-scale PatchGAN Discriminator with auxiliary classification.

Implements the discriminator from:
    "Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"
    (arXiv: 2103.15158)

The discriminator operates at two scales (original + 0.5× downsampled) and
each scale produces:
    - An adversarial patch map (real/fake per receptive-field patch).
    - Auxiliary class logits for defect category classification.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single-scale PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with auxiliary classification branch.

    Architecture::

        Layer 0: Conv4↓(3 →  64)                + LeakyReLU(0.2)
        Layer 1: Conv4↓(64 → 128) + InstanceNorm + LeakyReLU(0.2)
        Layer 2: Conv4↓(128→ 256) + InstanceNorm + LeakyReLU(0.2)
        Layer 3: Conv4  (256→ 512) + InstanceNorm + LeakyReLU(0.2)  (stride=1)
        Layer 4: Conv4  (512→   1)                                   (stride=1) → adversarial

        Auxiliary branch (from 256-channel features after Layer 2):
            GlobalAvgPool → Linear(256, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 1,
    ) -> None:
        """Initialise single-scale PatchGAN.

        Args:
            in_channels: Input image channels (3 for RGB).
            base_channels: Base channel width.
            num_classes: Number of defect categories for the auxiliary head.
        """
        super().__init__()
        assert in_channels == 3, "Discriminator only supports 3-channel RGB input."

        ch = base_channels

        # Main discriminator pathway
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # → [B, 64, H/2, W/2]

        self.layer1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  # → [B, 128, H/4, W/4]

        self.layer2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # → [B, 256, H/8, W/8]  ← auxiliary branch taps here

        self.layer3 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # → [B, 512, H/8-1, W/8-1]  (stride=1, reduces by kernel-1)

        # Adversarial output (no activation — raw logits for loss computation)
        self.adv_head = nn.Conv2d(ch * 8, 1, kernel_size=4, stride=1, padding=1)
        # → [B, 1, patch_h, patch_w]

        # Auxiliary classification branch from 256-channel features
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # [B, 256, 1, 1]
            nn.Flatten(),                      # [B, 256]
            nn.Linear(ch * 4, num_classes),    # [B, num_classes]
        )

        # Weight initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """He init for conv layers, Xavier for linear layers."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminate a single-scale image.

        Args:
            x: Input RGB image [B, 3, H, W].

        Returns:
            Tuple of:
                - adv_output: Adversarial patch logits [B, 1, patch_h, patch_w].
                - cls_logits: Auxiliary class logits [B, num_classes].
        """
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)   # 256-channel features
        h3 = self.layer3(h2)

        adv_output = self.adv_head(h3)
        cls_logits = self.aux_head(h2)

        return adv_output, cls_logits


# ---------------------------------------------------------------------------
# Multi-scale Discriminator
# ---------------------------------------------------------------------------

class MultiscaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator (2 scales).

    Processes the input at original resolution and at 0.5× downsampled
    resolution.  Each scale has an independent ``PatchGANDiscriminator``.

    Returns a list of ``(adv_output, cls_logits)`` tuples, one per scale.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 1,
        num_scales: int = 2,
    ) -> None:
        """Initialise multi-scale discriminator.

        Args:
            in_channels: Input image channels (3 for RGB).
            base_channels: Base channel width per PatchGAN.
            num_classes: Number of defect categories for auxiliary heads.
            num_scales: Number of discriminator scales (default 2).
        """
        super().__init__()
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, base_channels, num_classes)
            for _ in range(num_scales)
        ])

        # Learnable downsampling is not used — we simply average-pool.
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    def forward(
        self, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run discriminators at each scale.

        Args:
            x: Input RGB image [B, 3, H, W].

        Returns:
            List of ``(adv_output, cls_logits)`` tuples.
            Index 0 = original scale, index 1 = 0.5× scale, etc.
        """
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        current = x

        for i, disc in enumerate(self.discriminators):
            results.append(disc(current))
            if i < self.num_scales - 1:
                current = self.downsample(current)

        return results


class StarGANDiscriminator(nn.Module):
    """Single shared raw PatchGAN critic with auxiliary image classification.

    Global pooling adapts the class head to native 1024px. Depth/channel cap
    are project settings; Defect-GAN refers to StarGAN without listing them.
    No normalization is used in the WGAN-GP critic.
    """

    def __init__(self, in_channels=3, base_channels=64, num_classes=1, num_layers=6):
        super().__init__()
        if in_channels != 3 or num_layers < 1:
            raise ValueError("Expected RGB and positive critic depth")
        layers = []
        current = in_channels
        for i in range(num_layers):
            channels = min(base_channels * 2**i, 2048)
            layers.extend([nn.Conv2d(current, channels, 4, 2, 1), nn.LeakyReLU(.01)])
            current = channels
        self.trunk = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.src = nn.Conv2d(current, 1, 3, 1, 1, bias=False)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(current, num_classes, bias=False))
        self.apply(PatchGANDiscriminator._init_weights)

    def forward(self, x):
        if min(x.shape[-2:]) < 2**self.num_layers:
            raise ValueError("Input too small for configured critic depth")
        h = self.trunk(x)
        return [(self.src(h), self.cls(h))]
