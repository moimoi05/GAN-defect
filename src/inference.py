"""Inference script for Defect-GAN RGB-to-RGB defect synthesis and restoration."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch

from src.models.defect_gan import DefectGAN
from src.utils import save_rgb_image, denormalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DefectGAN.Inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defect-GAN RGB Inference Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to single input RGB image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output RGB image.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory of RGB images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated RGB images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["defacement", "restoration"],
        default="defacement",
        help="Translation mode: defacement (OK->NG) or restoration (NG->OK).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1024,
        help="Patch size for processing large images (default: 1024).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=768,
        help="Stride for sliding window inference on large images (default: 768).",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the internal alpha/spatial composition mask.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on (cuda, cpu, auto).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> DefectGAN:
    """Load model architecture and weights from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    gen_cfg = model_cfg.get("generator", {})
    disc_cfg = model_cfg.get("discriminator", {})

    epoch = checkpoint.get("epoch", 0)
    if epoch == 0:
        logger.warning(
            "Checkpoint indicates epoch 0! Weights may be untrained. Outputs will reflect untrained initialization."
        )

    model = DefectGAN(
        in_channels=3,
        base_channels_g=gen_cfg.get("base_channels", 64),
        base_channels_d=disc_cfg.get("base_channels", 64),
        num_classes=gen_cfg.get("num_classes", 1),
        num_res_blocks=gen_cfg.get("num_resblocks", 6),
        num_disc_scales=disc_cfg.get("num_scales", 2),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def process_patch(
    model: DefectGAN,
    patch: torch.Tensor,
    mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on a single 3-channel patch tensor [1, 3, H, W] in [-1, 1]."""
    c = torch.ones(1, model.num_classes, device=device)
    with torch.no_grad():
        if mode == "defacement":
            res = model.defacement(patch, c)
        else:
            res = model.restoration(patch, c)

    return res["output"], res["mask"]


def run_sliding_window_inference(
    model: DefectGAN,
    image_tensor: torch.Tensor,
    mode: str,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process an arbitrarily large image using blended sliding window patches."""
    _, _, h, w = image_tensor.shape

    # If image matches patch size exactly, run directly
    if h == patch_size and w == patch_size:
        return process_patch(model, image_tensor, mode, device)

    # Accumulators for blended result
    out_accum = torch.zeros((1, 3, h, w), device=device)
    mask_accum = torch.zeros((1, 1, h, w), device=device)
    weight_accum = torch.zeros((1, 1, h, w), device=device)

    # 2D Hann window for smooth seam blending
    hann_1d = torch.hann_window(patch_size, device=device)
    window_2d = torch.outer(hann_1d, hann_1d).unsqueeze(0).unsqueeze(0) + 1e-4

    y_steps = max(1, math.ceil((h - patch_size) / stride) + 1) if h > patch_size else 1
    x_steps = max(1, math.ceil((w - patch_size) / stride) + 1) if w > patch_size else 1

    for i in range(y_steps):
        y0 = min(i * stride, h - patch_size)
        y1 = y0 + patch_size
        for j in range(x_steps):
            x0 = min(j * stride, w - patch_size)
            x1 = x0 + patch_size

            patch = image_tensor[:, :, y0:y1, x0:x1]
            out_patch, mask_patch = process_patch(model, patch, mode, device)

            out_accum[:, :, y0:y1, x0:x1] += out_patch * window_2d
            mask_accum[:, :, y0:y1, x0:x1] += mask_patch * window_2d
            weight_accum[:, :, y0:y1, x0:x1] += window_2d

    out_final = torch.clamp(out_accum / weight_accum, -1.0, 1.0)
    mask_final = torch.clamp(mask_accum / weight_accum, 0.0, 1.0)
    return out_final, mask_final


def process_single_image(
    model: DefectGAN,
    input_path: Path,
    output_path: Path,
    mode: str,
    patch_size: int,
    stride: int,
    save_mask: bool,
    device: torch.device,
) -> None:
    """Run inference on a single RGB image file."""
    with Image.open(input_path) as img:
        img_rgb = img.convert("RGB")

    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)  # [1, 3, H, W] in [-1, 1]

    out_tensor, mask_tensor = run_sliding_window_inference(
        model=model,
        image_tensor=tensor,
        mode=mode,
        patch_size=patch_size,
        stride=stride,
        device=device,
    )

    save_rgb_image(out_tensor.squeeze(0), output_path)
    logger.info(f"Saved generated image: {output_path}")

    if save_mask:
        mask_path = output_path.parent / f"{output_path.stem}_mask.png"
        save_rgb_image(mask_tensor.squeeze(0), mask_path)
        logger.info(f"Saved internal mask: {mask_path}")


def main() -> None:
    args = parse_args()

    dev_str = args.device
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)
    logger.info(f"Inference device: {device}")

    model = load_model(Path(args.checkpoint), device)

    # Single image mode
    if args.input:
        if not args.output:
            raise ValueError("--output must be specified when using --input")
        process_single_image(
            model=model,
            input_path=Path(args.input),
            output_path=Path(args.output),
            mode=args.mode,
            patch_size=args.patch_size,
            stride=args.stride,
            save_mask=args.save_mask,
            device=device,
        )

    # Directory batch mode
    elif args.input_dir:
        if not args.output_dir:
            raise ValueError("--output-dir must be specified when using --input-dir")

        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        image_files = [p for p in in_dir.glob("*") if p.suffix.lower() in valid_exts]

        logger.info(f"Found {len(image_files)} images to process in {in_dir}")

        for img_p in image_files:
            out_p = out_dir / f"{img_p.stem}_synth_{args.mode}.png"
            process_single_image(
                model=model,
                input_path=img_p,
                output_path=out_p,
                mode=args.mode,
                patch_size=args.patch_size,
                stride=args.stride,
                save_mask=args.save_mask,
                device=device,
            )

        logger.info(f"Completed processing all {len(image_files)} images.")

    else:
        raise ValueError("Either --input or --input-dir must be specified.")


if __name__ == "__main__":
    main()
