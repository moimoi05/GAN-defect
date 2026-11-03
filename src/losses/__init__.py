"""Losses package for Defect-GAN."""

from src.losses.gan_losses import (
    LSGANLoss,
    ClassificationLoss,
    CycleConsistencyLoss,
    IdentityLoss,
    MaskSparsityLoss,
    DefectGANLoss,
)

__all__ = [
    "LSGANLoss",
    "ClassificationLoss",
    "CycleConsistencyLoss",
    "IdentityLoss",
    "MaskSparsityLoss",
    "DefectGANLoss",
]
