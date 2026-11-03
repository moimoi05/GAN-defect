"""Evaluator module for Defect-GAN validation and image sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import DefectGANLoss
from src.utils import save_rgb_image, denormalize


class DefectGANEvaluator:
    """Evaluates Defect-GAN during validation and saves RGB sample visualizations."""

    def __init__(
        self,
        model: DefectGAN,
        criterion: DefectGANLoss,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self, epoch: int, save_samples: bool = True) -> Dict[str, float]:
        """Run validation loop over val_loader and optionally save preview RGB samples."""
        self.model.eval()

        val_metrics: Dict[str, float] = {
            "val_cyc_normal": 0.0,
            "val_cyc_defect": 0.0,
            "val_mask_defect_mean": 0.0,
            "val_mask_normal_mean": 0.0,
        }
        total_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            normal = batch["normal"].to(self.device)
            defect = batch["defect"].to(self.device)
            b_size = normal.size(0)

            c = torch.ones(b_size, self.model.num_classes, device=self.device)

            # Cycle translations
            fake_defect = self.model.defacement(normal, c)
            recon_normal = self.model.restoration(fake_defect["output"], c)

            fake_normal = self.model.restoration(defect, c)
            recon_defect = self.model.defacement(fake_normal["output"], c)

            # Background preservation & cycle metrics
            l1_norm = torch.mean(torch.abs(recon_normal["output"] - normal)).item()
            l1_def = torch.mean(torch.abs(recon_defect["output"] - defect)).item()
            mask_def_mean = torch.mean(fake_defect["mask"]).item()
            mask_norm_mean = torch.mean(fake_normal["mask"]).item()

            val_metrics["val_cyc_normal"] += l1_norm
            val_metrics["val_cyc_defect"] += l1_def
            val_metrics["val_mask_defect_mean"] += mask_def_mean
            val_metrics["val_mask_normal_mean"] += mask_norm_mean
            total_batches += 1

            # Save visual samples for the first batch
            if save_samples and batch_idx == 0:
                epoch_sample_dir = self.samples_dir / f"epoch_{epoch:04d}"
                epoch_sample_dir.mkdir(parents=True, exist_ok=True)

                for i in range(min(b_size, 4)):
                    save_rgb_image(normal[i], epoch_sample_dir / f"sample_{i}_01_normal_input.png")
                    save_rgb_image(fake_defect["output"][i], epoch_sample_dir / f"sample_{i}_02_fake_defect.png")
                    save_rgb_image(fake_defect["mask"][i], epoch_sample_dir / f"sample_{i}_03_defect_mask.png")
                    save_rgb_image(recon_normal["output"][i], epoch_sample_dir / f"sample_{i}_04_recon_normal.png")

                    save_rgb_image(defect[i], epoch_sample_dir / f"sample_{i}_05_defect_input.png")
                    save_rgb_image(fake_normal["output"][i], epoch_sample_dir / f"sample_{i}_06_fake_normal.png")
                    save_rgb_image(recon_defect["output"][i], epoch_sample_dir / f"sample_{i}_07_recon_defect.png")

        # Average metrics
        if total_batches > 0:
            for k in val_metrics:
                val_metrics[k] /= total_batches

        return val_metrics
