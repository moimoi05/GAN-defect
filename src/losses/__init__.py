"""Losses package for Defect-GAN."""

from src.losses.gan_losses import (
    LSGANLoss,
    WGANLoss,
    PaperDefectGANLoss,
    ClassificationLoss,
    CycleConsistencyLoss,
    IdentityLoss,
    MaskSparsityLoss,
    DefectGANLoss,
)

__all__ = [
    "LSGANLoss",
    "WGANLoss",
    "PaperDefectGANLoss",
    "ClassificationLoss",
    "CycleConsistencyLoss",
    "IdentityLoss",
    "MaskSparsityLoss",
    "DefectGANLoss",
]
