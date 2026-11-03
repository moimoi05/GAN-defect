"""Defect-GAN full model wrapper.

Provides a paper-inspired shared-network baseline and a separate legacy
dual-network variant, based on:
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
from src.models.discriminator import MultiscaleDiscriminator, StarGANDiscriminator


class DefectGAN(nn.Module):
    """Complete Defect-GAN model with bidirectional translation.

    Directional names alias the same generator and critic in the paper
    baseline; the legacy variant retains separate networks and checkpoints.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels_g: int = 64,
        base_channels_d: int = 64,
        num_classes: int = 1,
        num_res_blocks: int = 6,
        num_disc_scales: int = 2,
        architecture: str = "defectgan_custom_legacy",
        num_disc_layers: int = 6,
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

        if architecture not in ("defectgan_paper_baseline", "defectgan_custom_legacy"):
            raise ValueError("Unknown DefectGAN architecture")
        self.architecture = architecture
        paper = architecture == "defectgan_paper_baseline"
        gen_options = {"encoder_stride": 4, "scalar_noise": True} if paper else {}
        # ---- Generators ----
        self.G_n2d = DefectGANGenerator(
            in_channels=in_channels,
            base_channels=base_channels_g,
            num_classes=num_classes,
            num_res_blocks=num_res_blocks,
            **gen_options,
        )
        self.G_d2n = self.G_n2d if paper else DefectGANGenerator(
            in_channels=in_channels,
            base_channels=base_channels_g,
            num_classes=num_classes,
            num_res_blocks=num_res_blocks,
            **gen_options,
        )

        # ---- Discriminators ----
        self.D_defect = StarGANDiscriminator(in_channels, base_channels_d, num_classes, num_disc_layers) if paper else MultiscaleDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels_d,
            num_classes=num_classes,
            num_scales=num_disc_scales,
        )
        self.D_normal = self.D_defect if paper else MultiscaleDiscriminator(
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
        spatial_map=None, noise_mode=None, generator=None,
    ) -> Dict[str, torch.Tensor]:
        """Translate a normal image to the defect domain.

        Args:
            normal_img: Normal-domain image [B, 3, H, W], range [-1, 1].
            c: Category vector [B, num_classes].

        Returns:
            Generator output dict with ``'output'``, ``'foreground'``,
            ``'mask'`` keys.
        """
        if c is None:
            c = normal_img.new_ones(normal_img.shape[0], self.num_classes)
        return self.G_n2d(normal_img, c, spatial_map=spatial_map, noise_mode=noise_mode, generator=generator)

    def restoration(
        self,
        defect_img: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        spatial_map=None, noise_mode=None, generator=None,
    ) -> Dict[str, torch.Tensor]:
        """Translate a defect image to the normal domain.

        Args:
            defect_img: Defect-domain image [B, 3, H, W], range [-1, 1].
            c: Category vector [B, num_classes].

        Returns:
            Generator output dict with ``'output'``, ``'foreground'``,
            ``'mask'`` keys.
        """
        if self.architecture == "defectgan_paper_baseline":
            c = defect_img.new_zeros(defect_img.shape[0], self.num_classes)
        return self.G_d2n(defect_img, c, spatial_map=spatial_map, noise_mode=noise_mode, generator=generator)

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
        fake_defect = self.defacement(normal_img, c)
        recon_normal = self.restoration(fake_defect["output"], c)
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
        fake_normal = self.restoration(defect_img, c)
        recon_defect = self.defacement(fake_normal["output"], c)
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
        fake_defect = self.defacement(normal_img, c)
        fake_normal = self.restoration(defect_img, c)

        # Cycle reconstructions
        recon_normal = self.restoration(fake_defect["output"], c)
        recon_defect = self.defacement(fake_normal["output"], c)

        return {
            "fake_defect": fake_defect,
            "fake_normal": fake_normal,
            "recon_normal": recon_normal,
            "recon_defect": recon_defect,
        }


    def generator_parameters(self):
        """Unique parameters even when direction aliases share weights."""
        return iter(dict.fromkeys([*self.G_n2d.parameters(), *self.G_d2n.parameters()]))

    def discriminator_parameters(self):
        return iter(dict.fromkeys([*self.D_defect.parameters(), *self.D_normal.parameters()]))
