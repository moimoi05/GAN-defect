"""Training Engine for Defect-GAN."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import DefectGANLoss
from src.engine.evaluator import DefectGANEvaluator
from src.utils import check_tensor_valid, set_seed

logger = logging.getLogger(__name__)


class DefectGANTrainer:
    """Trainer orchestrator for Defect-GAN."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: DefectGAN,
        criterion: DefectGANLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)

        # Output subdirectories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = DefectGANEvaluator(
            model=self.model,
            criterion=self.criterion,
            val_loader=self.val_loader,
            device=self.device,
            output_dir=self.output_dir,
        )

        training_cfg = config.get("training", {})
        opt_cfg = training_cfg.get("optimizer", {})

        # Optimizers
        lr_g = opt_cfg.get("lr_g", 0.0001)
        lr_d = opt_cfg.get("lr_d", 0.0004)
        beta1 = opt_cfg.get("beta1", 0.0)
        beta2 = opt_cfg.get("beta2", 0.999)
        weight_decay = opt_cfg.get("weight_decay", 0.0001)

        # Generator parameters: G_n2d and G_d2n
        g_params = list(self.model.G_n2d.parameters()) + list(self.model.G_d2n.parameters())
        self.opt_g = torch.optim.Adam(
            g_params, lr=lr_g, betas=(beta1, beta2), weight_decay=weight_decay
        )

        # Discriminator parameters: D_defect and D_normal
        d_params = list(self.model.D_defect.parameters()) + list(self.model.D_normal.parameters())
        self.opt_d = torch.optim.Adam(
            d_params, lr=lr_d, betas=(beta1, beta2), weight_decay=weight_decay
        )

        # Gradient accumulation
        self.grad_accum_steps = max(1, training_cfg.get("gradient_accumulation_steps", 1))

        # Mixed precision
        amp_mode = training_cfg.get("mixed_precision", "auto")
        self.use_amp = (
            device.type == "cuda"
            and amp_mode in ("auto", "fp16", "bf16")
            and torch.cuda.is_available()
        )
        self.scaler_g = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scaler_d = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Schedulers
        self.epochs = training_cfg.get("epochs", 100)
        sched_cfg = training_cfg.get("scheduler", {})
        decay_epoch = sched_cfg.get("decay_start_epoch", self.epochs // 2)

        def lr_lambda(epoch: int) -> float:
            if epoch < decay_epoch:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_epoch) / (self.epochs - decay_epoch + 1e-8))

        if sched_cfg.get("type") == "linear_decay":
            self.sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, lr_lambda=lr_lambda)
            self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, lr_lambda=lr_lambda)
        else:
            self.sched_g = None
            self.sched_d = None

        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _set_requires_grad(self, nets: list, requires_grad: bool = False) -> None:
        """Helper to toggle gradients for freezing/unfreezing networks."""
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def train_step(
        self, normal: torch.Tensor, defect: torch.Tensor, step_in_epoch: int
    ) -> Dict[str, float]:
        """Execute a single Defect-GAN training iteration (D step + G step)."""
        b_size = normal.size(0)
        c = torch.ones(b_size, self.model.num_classes, device=self.device)

        target_defect = torch.ones(b_size, self.model.num_classes, device=self.device)
        target_normal = torch.zeros(b_size, self.model.num_classes, device=self.device)

        # ===================================================================
        # 1. Train Discriminators (D_defect, D_normal)
        # ===================================================================
        self._set_requires_grad([self.model.D_defect, self.model.D_normal], True)
        self._set_requires_grad([self.model.G_n2d, self.model.G_d2n], False)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Forward G to get fake images (detached for D training)
            with torch.no_grad():
                fake_defect = self.model.defacement(normal, c)
                fake_normal = self.model.restoration(defect, c)

            # Discriminate real images
            d_defect_real = self.model.discriminate_defect(defect)
            d_normal_real = self.model.discriminate_normal(normal)

            # Discriminate fake images (detached)
            d_defect_fake = self.model.discriminate_defect(fake_defect["output"].detach())
            d_normal_fake = self.model.discriminate_normal(fake_normal["output"].detach())

            # Compute D loss
            d_loss, d_logs = self.criterion.compute_discriminator_loss(
                d_defect_real=d_defect_real,
                d_defect_fake=d_defect_fake,
                d_normal_real=d_normal_real,
                d_normal_fake=d_normal_fake,
                target_cls_defect=target_defect,
                target_cls_normal=target_normal,
            )
            d_loss_scaled = d_loss / self.grad_accum_steps

        # Backward D
        self.scaler_d.scale(d_loss_scaled).backward()

        if (step_in_epoch + 1) % self.grad_accum_steps == 0:
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
            self.opt_d.zero_grad()

        # ===================================================================
        # 2. Train Generators (G_n2d, G_d2n)
        # ===================================================================
        self._set_requires_grad([self.model.D_defect, self.model.D_normal], False)
        self._set_requires_grad([self.model.G_n2d, self.model.G_d2n], True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Forward translations
            fake_defect_g = self.model.defacement(normal, c)
            fake_normal_g = self.model.restoration(defect, c)

            # Cycle reconstructions
            recon_normal = self.model.restoration(fake_defect_g["output"], c)
            recon_defect = self.model.defacement(fake_normal_g["output"], c)

            # Identity / reconstruction (optional if lambda_rec > 0)
            if self.criterion.lambda_rec > 0:
                id_defect = self.model.defacement(defect, c)
                id_normal = self.model.restoration(normal, c)
            else:
                id_defect, id_normal = None, None

            # Discriminators evaluate fake images (gradients flow back to G)
            d_defect_fake_g = self.model.discriminate_defect(fake_defect_g["output"])
            d_normal_fake_g = self.model.discriminate_normal(fake_normal_g["output"])

            # Compute G loss
            g_loss, g_logs = self.criterion.compute_generator_loss(
                normal_img=normal,
                defect_img=defect,
                fake_defect=fake_defect_g,
                fake_normal=fake_normal_g,
                recon_normal=recon_normal,
                recon_defect=recon_defect,
                id_defect=id_defect,
                id_normal=id_normal,
                d_defect_fake=d_defect_fake_g,
                d_normal_fake=d_normal_fake_g,
                target_cls_defect=target_defect,
                target_cls_normal=target_normal,
            )
            g_loss_scaled = g_loss / self.grad_accum_steps

        # Backward G
        self.scaler_g.scale(g_loss_scaled).backward()

        if (step_in_epoch + 1) % self.grad_accum_steps == 0:
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            self.opt_g.zero_grad()

        # Combine logs
        step_logs = {**d_logs, **g_logs}
        return step_logs

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save a complete training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "opt_g_state": self.opt_g.state_dict(),
            "opt_d_state": self.opt_d.state_dict(),
            "scaler_g_state": self.scaler_g.state_dict() if self.use_amp else None,
            "scaler_d_state": self.scaler_d.state_dict() if self.use_amp else None,
            "sched_g_state": self.sched_g.state_dict() if self.sched_g else None,
            "sched_d_state": self.sched_d.state_dict() if self.sched_d else None,
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from a checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.opt_g.load_state_dict(checkpoint["opt_g_state"])
        self.opt_d.load_state_dict(checkpoint["opt_d_state"])

        if self.use_amp and checkpoint.get("scaler_g_state"):
            self.scaler_g.load_state_dict(checkpoint["scaler_g_state"])
            self.scaler_d.load_state_dict(checkpoint["scaler_d_state"])

        if self.sched_g and checkpoint.get("sched_g_state"):
            self.sched_g.load_state_dict(checkpoint["sched_g_state"])
            self.sched_d.load_state_dict(checkpoint["sched_d_state"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def train(self, dry_run: bool = False) -> None:
        """Run training loop."""
        logger.info("Starting Defect-GAN training...")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")

        training_cfg = self.config.get("training", {})
        val_interval = training_cfg.get("validation_interval", 1)
        ckpt_interval = training_cfg.get("checkpoint_interval", 10)
        sample_interval = training_cfg.get("sample_interval", 5)

        start_epoch = self.current_epoch
        num_epochs = 1 if dry_run else self.epochs

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            step_count = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

            try:
                for step, batch in enumerate(pbar):
                    normal = batch["normal"].to(self.device)
                    defect = batch["defect"].to(self.device)

                    logs = self.train_step(normal, defect, step)

                    epoch_d_loss += logs["d_loss"]
                    epoch_g_loss += logs["g_loss"]
                    step_count += 1
                    self.global_step += 1

                    pbar.set_postfix({
                        "D_loss": f"{logs['d_loss']:.4f}",
                        "G_loss": f"{logs['g_loss']:.4f}",
                        "Cyc": f"{logs['g_cyc_normal']:.4f}",
                        "Mask": f"{logs['g_mask_defect']:.4f}",
                    })

                    if dry_run and step >= 1:
                        logger.info("Dry-run limit reached (2 steps). Exiting training loop.")
                        break

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA Out-Of-Memory encountered at epoch {epoch}, step {self.global_step}!")
                torch.cuda.empty_cache()
                emergency_ckpt = self.checkpoint_dir / f"emergency_oom_epoch_{epoch}.pth"
                self.save_checkpoint(emergency_ckpt)
                raise e

            # Step schedulers
            if self.sched_g:
                self.sched_g.step()
            if self.sched_d:
                self.sched_d.step()

            avg_d_loss = epoch_d_loss / max(1, step_count)
            avg_g_loss = epoch_g_loss / max(1, step_count)
            logger.info(
                f"Epoch {epoch} Finished: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}"
            )

            # Validation
            if (epoch + 1) % val_interval == 0 or dry_run:
                should_save_samples = (epoch + 1) % sample_interval == 0 or dry_run
                val_metrics = self.evaluator.evaluate(epoch, save_samples=should_save_samples)
                logger.info(
                    f"Epoch {epoch} Val Metrics: Cyc_N={val_metrics['val_cyc_normal']:.4f}, "
                    f"Cyc_D={val_metrics['val_cyc_defect']:.4f}, "
                    f"Mask_Def={val_metrics['val_mask_defect_mean']:.4f}"
                )

                # Check if best
                total_val_loss = val_metrics["val_cyc_normal"] + val_metrics["val_cyc_defect"]
                is_best = total_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = total_val_loss

            # Save checkpoint
            if (epoch + 1) % ckpt_interval == 0 or dry_run or epoch == num_epochs - 1:
                ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
                self.save_checkpoint(ckpt_path, is_best=False)

            if dry_run:
                break

        logger.info("Training cycle finished successfully.")
