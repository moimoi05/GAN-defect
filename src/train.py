"""Main training entry point for Defect-GAN."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.f971_dataset import DefectGANDataset, verify_manifest_consistency
from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import DefectGANLoss, PaperDefectGANLoss
from src.engine.trainer import DefectGANTrainer
from src.utils import set_seed, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DefectGAN.Train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defect-GAN RGB Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defectgan_1024.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute a bounded smoke epoch proving actual G/D optimizer updates.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Alias for dry-run verification mode.",
    )
    parser.add_argument("--max-epochs-this-run", type=int, default=None,
                        help="Bound one invocation; useful for reviewed epoch-by-epoch pilots.")
    parser.add_argument("--verify-resume-only", action="store_true",
                        help="Load a trusted checkpoint and report restored state without training.")
    parser.add_argument(
        "--device",

        type=str,
        default=None,
        help="Override device (cuda, cpu, auto).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def bounded_smoke_dataset(dataset, config: dict, batch_size: int):
    """Use a complete small loader epoch; its checkpoint is smoke-only."""
    training = config.setdefault("training", {})
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    critic_steps = int(training.get("n_critic", 1))
    if min(batch_size, accumulation, critic_steps) < 1:
        raise ValueError("batch size, accumulation and n_critic must be positive")
    count = min(len(dataset), batch_size * accumulation * critic_steps)
    training["smoke_subset_samples"] = count
    return Subset(dataset, range(count))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    dry_run = args.dry_run or args.smoke_test

    # 1. Random seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Initialized random seed: {seed}")

    # 2. Output directory
    output_cfg = config.get("output", {})
    output_root = Path(os.environ.get("OUTPUT_DIR", output_cfg.get("root", "outputs")))
    if dry_run:
        output_root = output_root / "_smoke"
    output_dir = output_root / config.get("experiment", {}).get("name", "defectgan_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / "train.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # 3. Device detection
    dev_str = args.device or config.get("training", {}).get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    logger.info(f"Selected computation device: {device}")
    if device.type == "cpu" and not dry_run:
        logger.warning(
            "Running on CPU without --dry-run! Full training on CPU is extremely slow and not recommended."
        )

    # 4. Dataset & DataLoaders
    data_cfg = config.get("data", {})
    data_root = Path(os.environ.get("DATA_DIR", data_cfg.get("data_root", ".")))
    img_size = data_cfg.get("image_size", 1024)

    train_manifest = verify_manifest_consistency(
        data_cfg.get("train_manifest", "training/manifests/train.csv"),
        data_root=data_root,
    )
    val_manifest = verify_manifest_consistency(
        data_cfg.get("val_manifest", "training/manifests/val.csv"),
        data_root=data_root,
    )

    train_dataset = DefectGANDataset(
        manifest_path=train_manifest,
        data_root=data_root,
        split="train",
        image_size=img_size,
    )
    val_dataset = DefectGANDataset(
        manifest_path=val_manifest,
        data_root=data_root,
        split="validation",
        image_size=img_size,
    )

    training_cfg = config.get("training", {})
    if training_cfg.get("cover_all_manifest_rows_per_epoch", False):
        from src.datasets.full_patch_coverage import FullPatchCoverageDataset
        train_dataset = FullPatchCoverageDataset(train_dataset)
        logger.info("Pilot coverage enabled: %d manifest patches per epoch (%d clean + %d defect)",
                    len(train_dataset), train_dataset.clean_count, train_dataset.defect_count)
    batch_size = training_cfg.get("batch_size", 1)
    if dry_run:
        config = {**config, "smoke_subset": True}
        train_dataset = bounded_smoke_dataset(train_dataset, config, batch_size)
        logger.info("Smoke-only bounded training subset: %d samples; checkpoint scope=smoke_subset", len(train_dataset))
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    num_workers = data_cfg.get("num_workers", 0) if device.type == "cpu" else data_cfg.get("num_workers", 4)
    pin_mem = data_cfg.get("pin_memory", False) and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    logger.info(f"Train Dataset: {len(train_dataset)} pairs | Val Dataset: {len(val_dataset)} pairs")

    # 5. Model construction
    model_cfg = config.get("model", {})
    gen_cfg = model_cfg.get("generator", {})
    disc_cfg = model_cfg.get("discriminator", {})

    base_ch_g = gen_cfg.get("base_channels", 64)
    num_res = gen_cfg.get("num_resblocks", 6)
    num_classes = gen_cfg.get("num_classes", 1)

    base_ch_d = disc_cfg.get("base_channels", 64)
    num_scales = disc_cfg.get("num_scales", 2)

    model = DefectGAN(
        in_channels=3,
        base_channels_g=base_ch_g,
        base_channels_d=base_ch_d,
        num_classes=num_classes,
        num_res_blocks=num_res,
        num_disc_scales=num_scales,
        architecture=model_cfg.get("architecture", "defectgan_paper_baseline"),
        num_disc_layers=disc_cfg.get("num_layers", 6),
    )

    params = count_parameters(model)
    logger.info(f"Defect-GAN Model Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")

    # 6. Loss function
    loss_weights = training_cfg.get("loss_weights", {})
    if model.architecture == "defectgan_paper_baseline":
        criterion = PaperDefectGANLoss(**loss_weights)
    else:
        criterion = DefectGANLoss(**loss_weights)

    # 7. Trainer
    trainer = DefectGANTrainer(
        config=config,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
    )

    # Resume if requested
    resume_path = args.resume or training_cfg.get("resume")
    if resume_path:
        trainer.load_checkpoint(Path(resume_path))
        logger.info("Checkpoint restored: epoch=%d G=%d D=%d ES=%s", trainer.current_epoch,
                    trainer.generator_updates, trainer.discriminator_updates,
                    trainer.early_stopping.state_dict())
    if args.verify_resume_only:
        if not resume_path:
            raise ValueError("--verify-resume-only requires --resume")
        return

    # 8. Run training
    logger.info("Executing training session...")
    trainer.train(dry_run=dry_run, max_epochs_this_run=args.max_epochs_this_run)
    logger.info("Training process completed.")


if __name__ == "__main__":
    main()
