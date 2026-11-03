"""Defect-GAN full model wrapper.

Combines dual generators (defacement & restoration) with dual discriminators
as described in:
    "Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"
    (arXiv: 2103.15158)

Usage::

    model = DefectGAN(num_classes=1)
    fake_defect = model.defacement(normal_img, c)
    fake_normal = model.restoration(defect_img, c)
    recon_normal, intermediates = model.cycle_defacement(normal_img, c)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.generator import DefectGANGenerator
from src.models.discriminator import MultiscaleDiscriminator


class DefectGAN(nn.Module):
    """Complete Defect-GAN model with bidirectional translation.

    Components:
        - ``G_n2d``: Generator for normal → defect (defacement).
        - ``G_d2n``: Generator for defect → normal (restoration).
        - ``D_defect``: Discriminator for the defect domain.
        - ``D_normal``: Discriminator for the normal domain.

    Both generators share the same architecture (``DefectGANGenerator``)
    but maintain **separate** weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels_g: int = 64,
        base_channels_d: int = 64,
        num_classes: int = 1,
        num_res_blocks: int = 6,
        num_disc_scales: int = 2,
    ) -> None:
        """Initialise the full Defect-GAN model.

        Args:
            in_channels: Image channels (must be 3 for RGB).
            base_channels_g: Base channel width for generators.
            base_channels_d: Base channel width for discriminators.
            num_classes: Number of defect categories (default 1 = scratch).
            num_res_blocks: Residual blocks in each generator bottleneck.
            num_disc_scales: Number of scales in each discriminator.
        """
        super().__init__()

        # ---- Generators (separate weights, same architecture) ----
        self.G_n2d = DefectGANGenerator(
            in_channels=in_channels,
            base_channels=base_channels_g,
            num_classes=num_classes,
            num_res_blocks=num_res_blocks,
        )
        self.G_d2n = DefectGANGenerator(
            in_channels=in_channels,
            base_channels=base_channels_g,
            num_classes=num_classes,
            num_res_blocks=num_res_blocks,
        )

        # ---- Discriminators ----
        self.D_defect = MultiscaleDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels_d,
            num_classes=num_classes,
            num_scales=num_disc_scales,
        )
        self.D_normal = MultiscaleDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels_d,
            num_classes=num_classes,
            num_scales=num_disc_scales,
        )

        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Core translation methods
    # ------------------------------------------------------------------

    def defacement(
        self,
        normal_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Translate a normal image to the defect domain.

        Args:
            normal_img: Normal-domain image [B, 3, H, W], range [-1, 1].
            c: Category vector [B, num_classes].

        Returns:
            Generator output dict with ``'output'``, ``'foreground'``,
            ``'mask'`` keys.
        """
        return self.G_n2d(normal_img, c)

    def restoration(
        self,
        defect_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Translate a defect image to the normal domain.

        Args:
            defect_img: Defect-domain image [B, 3, H, W], range [-1, 1].
            c: Category vector [B, num_classes].

        Returns:
            Generator output dict with ``'output'``, ``'foreground'``,
            ``'mask'`` keys.
        """
        return self.G_d2n(defect_img, c)

    # ------------------------------------------------------------------
    # Cycle-consistency methods
    # ------------------------------------------------------------------

    def cycle_defacement(
        self,
        normal_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Full cycle: normal → fake_defect → reconstructed_normal.

        Args:
            normal_img: Normal-domain image [B, 3, H, W].
            c: Category vector [B, num_classes].

        Returns:
            Tuple of:
                - ``recon_normal``: Reconstruction dict from G_d2n
                  (contains ``'output'``, ``'foreground'``, ``'mask'``).
                - ``fake_defect``: Intermediate defacement dict from G_n2d.
        """
        fake_defect = self.G_n2d(normal_img, c)
        recon_normal = self.G_d2n(fake_defect["output"], c)
        return recon_normal, fake_defect

    def cycle_restoration(
        self,
        defect_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Full cycle: defect → fake_normal → reconstructed_defect.

        Args:
            defect_img: Defect-domain image [B, 3, H, W].
            c: Category vector [B, num_classes].

        Returns:
            Tuple of:
                - ``recon_defect``: Reconstruction dict from G_n2d
                  (contains ``'output'``, ``'foreground'``, ``'mask'``).
                - ``fake_normal``: Intermediate restoration dict from G_d2n.
        """
        fake_normal = self.G_d2n(defect_img, c)
        recon_defect = self.G_n2d(fake_normal["output"], c)
        return recon_defect, fake_normal

    # ------------------------------------------------------------------
    # Discriminator convenience methods
    # ------------------------------------------------------------------

    def discriminate_defect(self, img: torch.Tensor):
        """Run the defect-domain discriminator.

        Args:
            img: Image to discriminate [B, 3, H, W].

        Returns:
            List of (adv_output, cls_logits) per scale.
        """
        return self.D_defect(img)

    def discriminate_normal(self, img: torch.Tensor):
        """Run the normal-domain discriminator.

        Args:
            img: Image to discriminate [B, 3, H, W].

        Returns:
            List of (adv_output, cls_logits) per scale.
        """
        return self.D_normal(img)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def forward(
        self,
        normal_img: torch.Tensor,
        defect_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Full forward pass for training (both directions + cycles).

        This is a convenience method that computes all translations needed
        for the training losses in a single call.

        Args:
            normal_img: Normal-domain batch [B, 3, H, W].
            defect_img: Defect-domain batch [B, 3, H, W].
            c: Category vector [B, num_classes].

        Returns:
            Dictionary of dictionaries::

                {
                    'fake_defect':  G_n2d(normal, c)           output dict,
                    'fake_normal':  G_d2n(defect, c)           output dict,
                    'recon_normal': G_d2n(fake_defect, c)      output dict,
                    'recon_defect': G_n2d(fake_normal, c)      output dict,
                }
        """
        # Forward translations
        fake_defect = self.G_n2d(normal_img, c)
        fake_normal = self.G_d2n(defect_img, c)

        # Cycle reconstructions
        recon_normal = self.G_d2n(fake_defect["output"], c)
        recon_defect = self.G_n2d(fake_normal["output"], c)

        return {
            "fake_defect": fake_defect,
            "fake_normal": fake_normal,
            "recon_normal": recon_normal,
            "recon_defect": recon_defect,
        }
