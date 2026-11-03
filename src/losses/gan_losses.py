"""Loss functions for Defect-GAN.

Implements all optimization objectives from:
    "Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"
    (arXiv: 2103.15158)

Loss components:
    1. Adversarial Loss (LSGAN / MSE): forces realism on generated images.
    2. Auxiliary Classification Loss: ensures correct defect category conditioning.
    3. Cycle Consistency Loss (L1): preserves background structure across full cycles.
    4. Reconstruction / Identity Loss (L1): preserves domain identity when input matches target.
    5. Mask Regularization Loss (L1): encourages sparsity in the internal alpha/spatial mask,
       strictly preserving flawless background areas.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import check_tensor_valid


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss (Mao et al., ICCV 2017).

    For Discriminator:
        L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[(D(fake))^2]
    For Generator:
        L_G = 0.5 * E[(D(fake) - 1)^2]
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def discriminator_loss(
        self,
        real_preds: List[torch.Tensor],
        fake_preds: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute LSGAN loss for discriminator across multi-scale outputs."""
        loss = 0.0
        for real_pred, fake_pred in zip(real_preds, fake_preds):
            target_real = torch.ones_like(real_pred)
            target_fake = torch.zeros_like(fake_pred)
            loss_scale = 0.5 * (self.mse(real_pred, target_real) + self.mse(fake_pred, target_fake))
            loss = loss + loss_scale
        return loss / len(real_preds)

    def generator_loss(self, fake_preds: List[torch.Tensor]) -> torch.Tensor:
        """Compute LSGAN loss for generator across multi-scale outputs."""
        loss = 0.0
        for fake_pred in fake_preds:
            target_real = torch.ones_like(fake_pred)
            loss_scale = 0.5 * self.mse(fake_pred, target_real)
            loss = loss + loss_scale
        return loss / len(fake_preds)


class ClassificationLoss(nn.Module):
    """Auxiliary Classification Loss for multi-scale discriminator.

    Uses BCEWithLogitsLoss for binary/single-class defect classification.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cls_logits_list: List[torch.Tensor],
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss averaged across scales."""
        loss = 0.0
        for logits in cls_logits_list:
            loss = loss + self.bce(logits, target_labels)
        return loss / len(cls_logits_list)


class CycleConsistencyLoss(nn.Module):
    """L1 Cycle Consistency Loss.

    L_cyc = || G_d2n(G_n2d(n)) - n ||_1 + || G_n2d(G_d2n(d)) - d ||_1
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.l1(reconstructed, original)


class IdentityLoss(nn.Module):
    """L1 Identity / Reconstruction Loss.

    Encourages mapping networks to act as identity when applied to target domain:
    || G_d2n(n) - n ||_1  and  || G_n2d(d) - d ||_1
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, transformed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.l1(transformed, original)


class MaskSparsityLoss(nn.Module):
    """Spatial Mask Regularization Loss.

    Enforces sparsity on the internal alpha/spatial mask M in [0, 1].
    Minimizes the modified area to preserve maximum background context.
    L_mask = E[ || M ||_1 ]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(mask))


class DefectGANLoss(nn.Module):
    """Composite Defect-GAN Loss manager with configurable hyperparameter weights."""

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_cyc: float = 10.0,
        lambda_rec: float = 10.0,
        lambda_mask: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cls = lambda_cls
        self.lambda_cyc = lambda_cyc
        self.lambda_rec = lambda_rec
        self.lambda_mask = lambda_mask

        self.adv_loss = LSGANLoss()
        self.cls_loss = ClassificationLoss()
        self.cyc_loss = CycleConsistencyLoss()
        self.id_loss = IdentityLoss()
        self.mask_loss = MaskSparsityLoss()

    def compute_discriminator_loss(
        self,
        d_defect_real: List[Tuple[torch.Tensor, torch.Tensor]],
        d_defect_fake: List[Tuple[torch.Tensor, torch.Tensor]],
        d_normal_real: List[Tuple[torch.Tensor, torch.Tensor]],
        d_normal_fake: List[Tuple[torch.Tensor, torch.Tensor]],
        target_cls_defect: torch.Tensor,
        target_cls_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total discriminator loss for D_defect and D_normal."""
        # Unpack predictions and logits
        defect_real_adv = [p[0] for p in d_defect_real]
        defect_real_cls = [p[1] for p in d_defect_real]
        defect_fake_adv = [p[0] for p in d_defect_fake]

        normal_real_adv = [p[0] for p in d_normal_real]
        normal_real_cls = [p[1] for p in d_normal_real]
        normal_fake_adv = [p[0] for p in d_normal_fake]

        # 1. Adversarial losses
        loss_d_defect_adv = self.adv_loss.discriminator_loss(defect_real_adv, defect_fake_adv)
        loss_d_normal_adv = self.adv_loss.discriminator_loss(normal_real_adv, normal_fake_adv)
        total_adv = (loss_d_defect_adv + loss_d_normal_adv) * self.lambda_adv

        # 2. Classification losses on real images
        loss_cls_defect = self.cls_loss(defect_real_cls, target_cls_defect)
        loss_cls_normal = self.cls_loss(normal_real_cls, target_cls_normal)
        total_cls = (loss_cls_defect + loss_cls_normal) * self.lambda_cls

        total_d_loss = total_adv + total_cls

        check_tensor_valid(total_d_loss, "Discriminator total loss")

        log_dict = {
            "d_loss": total_d_loss.item(),
            "d_adv_defect": loss_d_defect_adv.item(),
            "d_adv_normal": loss_d_normal_adv.item(),
            "d_cls_defect": loss_cls_defect.item(),
            "d_cls_normal": loss_cls_normal.item(),
        }
        return total_d_loss, log_dict

    def compute_generator_loss(
        self,
        normal_img: torch.Tensor,
        defect_img: torch.Tensor,
        fake_defect: Dict[str, torch.Tensor],
        fake_normal: Dict[str, torch.Tensor],
        recon_normal: Dict[str, torch.Tensor],
        recon_defect: Dict[str, torch.Tensor],
        id_defect: Optional[Dict[str, torch.Tensor]],
        id_normal: Optional[Dict[str, torch.Tensor]],
        d_defect_fake: List[Tuple[torch.Tensor, torch.Tensor]],
        d_normal_fake: List[Tuple[torch.Tensor, torch.Tensor]],
        target_cls_defect: torch.Tensor,
        target_cls_normal: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total generator loss for G_n2d and G_d2n."""
        defect_fake_adv = [p[0] for p in d_defect_fake]
        defect_fake_cls = [p[1] for p in d_defect_fake]
        normal_fake_adv = [p[0] for p in d_normal_fake]
        normal_fake_cls = [p[1] for p in d_normal_fake]

        # 1. Adversarial loss (fool discriminators)
        loss_g_defect_adv = self.adv_loss.generator_loss(defect_fake_adv)
        loss_g_normal_adv = self.adv_loss.generator_loss(normal_fake_adv)
        total_adv = (loss_g_defect_adv + loss_g_normal_adv) * self.lambda_adv

        # 2. Classification loss on generated images
        loss_cls_defect = self.cls_loss(defect_fake_cls, target_cls_defect)
        loss_cls_normal = self.cls_loss(normal_fake_cls, target_cls_normal)
        total_cls = (loss_cls_defect + loss_cls_normal) * self.lambda_cls

        # 3. Cycle consistency loss
        loss_cyc_normal = self.cyc_loss(recon_normal["output"], normal_img)
        loss_cyc_defect = self.cyc_loss(recon_defect["output"], defect_img)
        total_cyc = (loss_cyc_normal + loss_cyc_defect) * self.lambda_cyc

        # 4. Identity / Reconstruction loss
        if id_defect is not None and id_normal is not None:
            loss_id_defect = self.id_loss(id_defect["output"], defect_img)
            loss_id_normal = self.id_loss(id_normal["output"], normal_img)
            total_id = (loss_id_defect + loss_id_normal) * self.lambda_rec
        else:
            loss_id_defect = torch.tensor(0.0, device=normal_img.device)
            loss_id_normal = torch.tensor(0.0, device=normal_img.device)
            total_id = torch.tensor(0.0, device=normal_img.device)

        # 5. Mask sparsity regularization (encourages minimal background modification)
        loss_mask_defect = self.mask_loss(fake_defect["mask"])
        loss_mask_normal = self.mask_loss(fake_normal["mask"])
        total_mask = (loss_mask_defect + loss_mask_normal) * self.lambda_mask

        # Total Generator Loss
        total_g_loss = total_adv + total_cls + total_cyc + total_id + total_mask

        check_tensor_valid(total_g_loss, "Generator total loss")

        log_dict = {
            "g_loss": total_g_loss.item(),
            "g_adv_defect": loss_g_defect_adv.item(),
            "g_adv_normal": loss_g_normal_adv.item(),
            "g_cls_defect": loss_cls_defect.item(),
            "g_cls_normal": loss_cls_normal.item(),
            "g_cyc_normal": loss_cyc_normal.item(),
            "g_cyc_defect": loss_cyc_defect.item(),
            "g_id_defect": loss_id_defect.item() if isinstance(loss_id_defect, torch.Tensor) else 0.0,
            "g_id_normal": loss_id_normal.item() if isinstance(loss_id_normal, torch.Tensor) else 0.0,
            "g_mask_defect": loss_mask_defect.item(),
            "g_mask_normal": loss_mask_normal.item(),
        }
        return total_g_loss, log_dict
