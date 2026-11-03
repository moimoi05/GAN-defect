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
from torch.utils.data import DataLoader

from src.datasets.f971_dataset import DefectGANDataset
from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import DefectGANLoss
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
        help="Execute 2 steps on CPU/GPU to verify code without full training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Alias for dry-run verification mode.",
    )
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
    output_dir = output_root / config.get("experiment", {}).get("name", "defectgan_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of config
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

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

    train_manifest = data_cfg.get("train_manifest", "training/manifests/train.csv")
    val_manifest = data_cfg.get("val_manifest", "training/manifests/val.csv")

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
    batch_size = training_cfg.get("batch_size", 1)
    num_workers = data_cfg.get("num_workers", 0) if device.type == "cpu" else data_cfg.get("num_workers", 4)
    pin_mem = data_cfg.get("pin_memory", False) and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
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
    )

    params = count_parameters(model)
    logger.info(f"Defect-GAN Model Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")

    # 6. Loss function
    loss_weights = training_cfg.get("loss_weights", {})
    criterion = DefectGANLoss(
        lambda_adv=loss_weights.get("lambda_adv", 1.0),
        lambda_cls=loss_weights.get("lambda_cls", 1.0),
        lambda_cyc=loss_weights.get("lambda_cyc", 10.0),
        lambda_rec=loss_weights.get("lambda_rec", 10.0),
        lambda_mask=loss_weights.get("lambda_mask", 0.1),
    )

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

    # 8. Run training
    logger.info("Executing training session...")
    trainer.train(dry_run=dry_run)
    logger.info("Training process completed.")


if __name__ == "__main__":
    main()
