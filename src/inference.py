"""RGB inference with CPU tile accumulation and explicit application controls.

SPADE control maps guide generation; ROI/protected maps are optional application
constraints applied after blending, not ground-truth segmentation targets.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.defect_gan import DefectGAN

logger = logging.getLogger("DefectGAN.Inference")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--input-dir")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=["defacement", "restoration"], default="defacement")
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--spatial-map", help="Single Scratch SPADE control map; grayscale [0,255].")
    parser.add_argument("--roi", help="Allowed modification region; grayscale [0,255].")
    parser.add_argument("--protect", help="Protected region; grayscale [0,255], 255 preserves RGB.")
    parser.add_argument("--save-alpha", "--save-mask", dest="save_mask", action="store_true",
                        help="Save internal alpha separately as *_alpha.png (debug only).")
    return parser.parse_args(argv)


def load_model(checkpoint_path: Path, device: torch.device) -> DefectGAN:
    """Load trusted checkpoints; architecture is explicit, old checkpoints stay legacy.

    Checkpoint files use pickle and must come from a trusted source. Legacy and
    shared-weight baseline state dictionaries are never silently converted.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    gen_cfg = model_cfg.get("generator", {})
    disc_cfg = model_cfg.get("discriminator", {})
    architecture = checkpoint.get("architecture", model_cfg.get("architecture", "defectgan_custom_legacy"))
    model = DefectGAN(
        in_channels=3,
        base_channels_g=gen_cfg.get("base_channels", 64),
        base_channels_d=disc_cfg.get("base_channels", 64),
        num_classes=gen_cfg.get("num_classes", 1),
        num_res_blocks=gen_cfg.get("num_resblocks", 6),
        num_disc_scales=disc_cfg.get("num_scales", 2),
        num_disc_layers=disc_cfg.get("num_layers", 6),
        architecture=architecture,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if checkpoint.get("epoch", 0) == 0:
        logger.warning("Epoch-zero checkpoint: outputs do not demonstrate trained Scratch synthesis.")
    return model.to(device).eval()


def process_patch(
    model: DefectGAN, patch: torch.Tensor, mode: str, device: torch.device,
    *, spatial_map=None, noise_mode="deterministic", generator=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Only this RGB patch and its control map are copied onto the compute device."""
    if mode not in {"defacement", "restoration"}:
        raise ValueError(f"Unknown inference mode: {mode}")
    patch = patch.to(device)
    paper = getattr(model, "architecture", None) == "defectgan_paper_baseline"
    category = torch.full((1, model.num_classes),
                          float(mode == "defacement" or not paper), device=device)
    control = None if spatial_map is None else spatial_map.to(device)
    with torch.inference_mode():
        translate = model.defacement if mode == "defacement" else model.restoration
        result = translate(patch, category, spatial_map=control,
                           noise_mode=noise_mode, generator=generator)
    output, alpha = result["output"].cpu().float(), result["mask"].cpu().float()
    if output.shape != patch.shape or alpha.shape != (1, 1, *patch.shape[-2:]):
        raise ValueError("Generator returned invalid RGB/alpha dimensions")
    if not torch.isfinite(output).all() or not torch.isfinite(alpha).all():
        raise ValueError("Generator returned non-finite RGB/alpha")
    return output, alpha


def _control_tensor(value, height, width, channels, name):
    if value is None:
        return None
    value = value.detach().to(device="cpu", dtype=torch.float32)
    if value.shape != (1, channels, height, width):
        raise ValueError(f"{name} must have shape [1,{channels},{height},{width}]")
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f"{name} must be finite and within [0,1]")
    return value


def _starts(length, patch_size, stride):
    return sorted(set(range(0, length - patch_size + 1, stride)) | {length - patch_size})


def run_sliding_window_inference(
    model: DefectGAN, image_tensor: torch.Tensor, mode: str, patch_size: int,
    stride: int, device: torch.device, *, seed: Optional[int] = None,
    noise_mode: str = "deterministic", spatial_map=None, allowed_roi=None,
    protected_roi=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend complete tiles on CPU; return CPU RGB/alpha at the original size.

    GPU storage is proportional to a tile. Host RAM still scales with image
    area; this is not a measured VRAM guarantee for actual 11K S942 images.
    Hann blending reduces seams but does not guarantee learned tile consistency.
    Optional ROI compositing preserves every tensor value outside allowed areas;
    returned alpha includes these application constraints when supplied.
    """
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("Require patch_size > 0 and 0 < stride <= patch_size")
    if noise_mode not in {"deterministic", "stochastic"}:
        raise ValueError("noise_mode must be deterministic or stochastic")
    if mode not in {"defacement", "restoration"}:
        raise ValueError("mode must be defacement or restoration")
    if image_tensor.ndim != 4 or image_tensor.shape[:2] != (1, 3):
        raise ValueError("Input must be [1,3,H,W] RGB")
    height, width = image_tensor.shape[-2:]
    if min(height, width) <= 0 or not torch.isfinite(image_tensor).all():
        raise ValueError("Input must have positive dimensions and finite RGB")
    source = image_tensor.detach().to(device="cpu", dtype=torch.float32)
    control = _control_tensor(spatial_map,height,width,model.num_classes,"spatial_map")
    roi = _control_tensor(allowed_roi,height,width,1,"allowed_roi")
    protect = _control_tensor(protected_roi,height,width,1,"protected_roi")
    padded_h, padded_w = max(height, patch_size), max(width, patch_size)
    padding = (0, padded_w-width, 0, padded_h-height)
    source_padded = F.pad(source, padding, mode="replicate")
    if control is not None:
        control = F.pad(control, padding, mode="replicate")
    out_accum = torch.zeros((1, 3, padded_h, padded_w))
    alpha_accum = torch.zeros((1, 1, padded_h, padded_w))
    weights = torch.zeros_like(alpha_accum)
    hann = torch.hann_window(patch_size, periodic=False)
    window = torch.outer(hann, hann)[None,None] + 1e-4
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for y in _starts(padded_h, patch_size, stride):
        for x in _starts(padded_w, patch_size, stride):
            region = (..., slice(y,y+patch_size), slice(x,x+patch_size))
            patch_control = None if control is None else control[region]
            output, alpha = process_patch(model, source_padded[region], mode, device,
                spatial_map=patch_control, noise_mode=noise_mode, generator=generator)
            out_accum[region] += output * window
            alpha_accum[region] += alpha * window
            weights[region] += window
    if not torch.all(weights > 0):
        raise RuntimeError("Sliding-window coverage failed")
    output = (out_accum / weights)[...,:height,:width].clamp(-1,1)
    alpha = (alpha_accum / weights)[...,:height,:width].clamp(0,1)
    if roi is not None or protect is not None:
        allowed = torch.ones_like(alpha) if roi is None else roi
        if protect is not None:
            allowed = allowed * (1 - protect)
        # torch.where makes preservation exact, independent of roundoff in blending.
        output = torch.where(allowed == 0, source, source + allowed * (output-source))
        alpha = alpha * allowed
    return output, alpha


def process_single_image(
    model: DefectGAN, input_path: Path, output_path: Path, mode: str,
    patch_size: int, stride: int, save_mask: bool, device: torch.device, **controls,
) -> None:
    """Export RGB PNG; effective debug alpha is a separate grayscale PNG."""
    alpha_path = output_path.with_name(f"{output_path.stem}_alpha.png")
    if input_path.resolve() == output_path.resolve() or (
        save_mask and input_path.resolve() == alpha_path.resolve()
    ):
        raise ValueError("Output/debug alpha must not overwrite input")
    with Image.open(input_path) as image:
        original = np.array(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(original.copy()).permute(2,0,1).float()[None] / 127.5 - 1
    output, alpha = run_sliding_window_inference(model,tensor,mode,patch_size,stride,device,**controls)
    rgb = ((output[0].permute(1,2,0)+1)*127.5).round().clamp(0,255).byte().numpy()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    # Always PNG, regardless of the requested suffix, to preserve protected RGB bytes.
    Image.fromarray(rgb).save(output_path, format="PNG")
    if save_mask:
        alpha_bytes = (alpha[0,0]*255).round().byte().numpy()
        Image.fromarray(alpha_bytes).save(alpha_path)
    logger.info("Saved RGB image: %s", output_path)


def _load_map(path):
    if path is None:
        return None
    with Image.open(path) as image:
        array = np.array(image.convert("L"),dtype=np.float32) / 255
    return torch.from_numpy(array)[None,None]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = load_model(Path(args.checkpoint),device)
    controls = dict(seed=args.seed,noise_mode=args.noise,spatial_map=_load_map(args.spatial_map),
                    allowed_roi=_load_map(args.roi),protected_roi=_load_map(args.protect))
    if args.input:
        if not args.output:
            raise ValueError("--output is required with --input")
        pairs = [(Path(args.input),Path(args.output))]
    else:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --input-dir")
        directory = Path(args.input_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Input directory not found: {directory}")
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in
                       {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
        if not files:
            raise ValueError("Input directory contains no supported images")
        pairs = [(p,Path(args.output_dir)/f"{p.stem}_synth_{args.mode}.png") for p in files]
    sources = {source.resolve() for source,_ in pairs}
    destinations = []
    for _,target in pairs:
        destinations.append(target.resolve())
        if args.save_mask:
            destinations.append(target.with_name(f"{target.stem}_alpha.png").resolve())
    if any(target in sources for target in destinations):
        raise ValueError("Output/debug alpha must not overwrite input")
    if len(set(destinations)) != len(destinations):
        raise ValueError("Output names collide; use unique input stems or separate directories")
    for source,target in pairs:
        process_single_image(model,source,target,args.mode,args.patch_size,args.stride,
                             args.save_mask,device,**controls)


if __name__ == "__main__":
    main()
