"""Defect-GAN model components.

This package implements the Defect-GAN architecture from
"Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"
(arXiv: 2103.15158).

Modules:
    generator: Encoder-decoder generator with SPADE, layer-wise composition,
               and adaptive noise injection.
    discriminator: Multi-scale PatchGAN discriminator with auxiliary classifier.
    defect_gan: Full Defect-GAN wrapper with dual generators and discriminators.
"""

from src.models.generator import DefectGANGenerator
from src.models.discriminator import MultiscaleDiscriminator
from src.models.defect_gan import DefectGAN

__all__ = [
    "DefectGANGenerator",
    "MultiscaleDiscriminator",
    "DefectGAN",
]
