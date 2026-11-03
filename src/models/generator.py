"""Defect-GAN Generator with layer-wise composition, SPADE, and adaptive noise.

Implements the encoder-decoder generator from:
    "Defect-GAN: High-Fidelity Defect Synthesis for Automated Defect Inspection"
    (arXiv: 2103.15158)

Key architectural features:
    1. Layer-wise composition: output = (1 - M) * input + M * F
       where F is the foreground (defect content) and M is the spatial mask.
    2. SPADE (Spatially-Adaptive Normalization) in the decoder, conditioned on
       a category vector for multi-class defect control.
    3. Adaptive noise injection after each conv block for stochastic variation.
"""

from __future__ import annotations

from contextvars import ContextVar

_noise_settings = ContextVar("defectgan_noise_settings", default=(None, None))

def compose_layers(input_rgb, foreground, mask):
    """Differentiable RGB composition (paper Eqs. 2 and 4)."""
    if input_rgb.ndim != 4 or input_rgb.shape[1] != 3 or foreground.shape != input_rgb.shape:
        raise ValueError("Composition requires matching [B,3,H,W] RGB tensors")
    if mask.shape != (input_rgb.shape[0], 1, *input_rgb.shape[2:]):
        raise ValueError("Composition mask must be [B,1,H,W]")
    return (1.0-mask)*input_rgb + mask*foreground

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptive Noise Injection
# ---------------------------------------------------------------------------

class AdaptiveNoiseInjection(nn.Module):
    """Injects Gaussian noise scaled by learnable per-channel weights.

    For each channel *c*, the output is:
        out[:, c] = x[:, c] + weight[c] * noise[:, c]
    where noise ~ N(0, 1).  Weights are initialised to zero so the layer
    is a no-op at the start of training and the network can gradually learn
    how much stochasticity to admit per channel.
    """

    def __init__(self, num_channels: int, scalar: bool = False) -> None:
        super().__init__()
        # Learnable per-channel scale, initialised to 0 (identity at init)
        self.weight = nn.Parameter(torch.zeros(1, 1 if scalar else num_channels, 1, 1))

    def forward(self, x: torch.Tensor, noise_mode=None, generator=None) -> torch.Tensor:
        context_mode, context_generator = _noise_settings.get()
        mode = noise_mode if noise_mode is not None else context_mode
        rng = generator if generator is not None else context_generator
        if mode not in (None, "deterministic", "stochastic"):
            raise ValueError("noise_mode must be deterministic or stochastic")
        enabled = self.training if mode is None else mode == "stochastic"
        if not enabled:
            return x
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)
        return x + self.weight * noise


# ---------------------------------------------------------------------------
# SPADE Normalization
# ---------------------------------------------------------------------------

class SPADENorm(nn.Module):
    """Spatially-Adaptive Denormalization (SPADE).

    Given a feature map *x* and a conditioning attribute map *attr_map*,
    SPADE first normalises *x* with Instance Normalization, then applies
    spatially-varying affine parameters (gamma, beta) predicted from
    *attr_map* via a small conv network.

    During training with a single defect class, the attribute map is simply
    the category vector repeated spatially.  During inference it can be a
    user-supplied spatial control map.

    Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
    Normalization", CVPR 2019.
    """

    def __init__(self, norm_channels: int, attr_channels: int, hidden_channels: int = 128) -> None:
        """Initialise SPADE layer.

        Args:
            norm_channels: Number of channels in the feature map to normalise.
            attr_channels: Number of channels in the conditioning attribute map
                           (equals ``num_classes`` when using category vectors).
            hidden_channels: Hidden layer width inside the SPADE parameter
                             prediction network.
        """
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(norm_channels, affine=False)

        # Small conv network: attr_map → (gamma, beta) with spatial dims
        self.shared = nn.Sequential(
            nn.Conv2d(attr_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_gamma = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, attr_map: torch.Tensor) -> torch.Tensor:
        """Apply SPADE normalisation.

        Args:
            x: Feature tensor [B, norm_channels, H, W].
            attr_map: Conditioning map [B, attr_channels, H', W'].  Will be
                      resized to match *x* if spatial dims differ.

        Returns:
            Modulated feature tensor [B, norm_channels, H, W].
        """
        # Resize attribute map to match feature spatial size
        if attr_map.shape[2:] != x.shape[2:]:
            attr_map = F.interpolate(
                attr_map, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        normalised = self.instance_norm(x)

        hidden = self.shared(attr_map)
        gamma = self.conv_gamma(hidden)  # [B, norm_channels, H, W]
        beta = self.conv_beta(hidden)    # [B, norm_channels, H, W]

        return normalised * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Residual Block (bottleneck)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with InstanceNorm and optional adaptive noise injection.

    Structure:
        x → Conv3x3 → IN → ReLU → NoiseInj → Conv3x3 → IN → (+ x) → ReLU → NoiseInj
    """

    def __init__(self, channels: int, use_noise: bool = True) -> None:
        """Initialise residual block.

        Args:
            channels: Number of input and output channels.
            use_noise: Whether to apply AdaptiveNoiseInjection after each
                       sub-block.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

        self.noise1 = AdaptiveNoiseInjection(channels) if use_noise else nn.Identity()
        self.noise2 = AdaptiveNoiseInjection(channels) if use_noise else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Feature tensor [B, C, H, W].

        Returns:
            Residual-connected output [B, C, H, W].
        """
        residual = x
        out = self.noise1(F.relu(self.in1(self.conv1(x)), inplace=True))
        out = self.in2(self.conv2(out))
        out = self.noise2(F.relu(out + residual, inplace=True))
        return out


# ---------------------------------------------------------------------------
# SPADE Decoder Block
# ---------------------------------------------------------------------------

class SPADEDecoderBlock(nn.Module):
    """Single decoder stage: Upsample → Conv → SPADE → ReLU → NoiseInj."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attr_channels: int,
    ) -> None:
        """Initialise SPADE decoder block.

        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            attr_channels: Channels of the SPADE conditioning map.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spade = SPADENorm(out_channels, attr_channels)
        self.noise = AdaptiveNoiseInjection(out_channels)

    def forward(self, x: torch.Tensor, attr_map: torch.Tensor) -> torch.Tensor:
        """Upsample, convolve, SPADE-normalise, activate, inject noise.

        Args:
            x: Feature tensor [B, in_channels, H, W].
            attr_map: SPADE conditioning map [B, attr_channels, H', W'].

        Returns:
            Decoded features [B, out_channels, 2H, 2W].
        """
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.spade(x, attr_map)
        x = F.relu(x, inplace=True)
        x = self.noise(x)
        return x


# ---------------------------------------------------------------------------
# Full Generator
# ---------------------------------------------------------------------------

class DefectGANGenerator(nn.Module):
    """Defect-GAN encoder-decoder generator with layer-wise composition.

    Architecture overview::

        Encoder (stride-8 downsampling):
            Conv7→64 → Conv4↓128 → Conv4↓256 → Conv4↓512

        Bottleneck:
            6 × ResBlock(512) with AdaptiveNoiseInjection

        Decoder (SPADE + upsample ×3):
            512→256 → 256→128 → 128→64

        Output heads:
            Foreground F: Conv7→3 + Tanh   (RGB, range [-1,1])
            Mask M:       Conv7→1 + Sigmoid (alpha, range [0,1])

        Composition:
            output = (1 - M) * input + M * F

    The generator preserves the background by learning a soft composition mask.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 1,
        num_res_blocks: int = 6,
        encoder_stride: int = 8,
        scalar_noise: bool = False,
    ) -> None:
        """Initialise the Defect-GAN generator.

        Args:
            in_channels: Input image channels (must be 3 for RGB).
            base_channels: Base channel width (doubled at each encoder stage).
            num_classes: Number of defect categories for SPADE conditioning.
                         Defaults to 1 (single scratch class).
            num_res_blocks: Number of residual blocks in the bottleneck.
        """
        super().__init__()
        assert in_channels == 3, "Generator only supports 3-channel RGB input."

        self.num_classes = num_classes
        if encoder_stride not in (4, 8):
            raise ValueError("encoder_stride must be 4 or 8")
        self.encoder_stride = encoder_stride
        ch = base_channels  # shorthand

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
        )  # → [B, 64, H, W]

        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )  # → [B, 128, H/2, W/2]

        self.enc3 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 4),
            nn.ReLU(inplace=True),
        )  # → [B, 256, H/4, W/4]

        self.enc4 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 8),
            nn.ReLU(inplace=True),
        )  # → [B, 512, H/8, W/8]

        # ---- Bottleneck (6 ResBlocks at 512 channels) ----
        self.bottleneck = nn.Sequential(
            *[ResBlock(ch * 8, use_noise=True) for _ in range(num_res_blocks)]
        )

        # ---- Decoder (SPADE) ----
        self.dec1 = SPADEDecoderBlock(ch * 8, ch * 4, attr_channels=num_classes)
        # 512→256, H/8→H/4
        self.dec2 = SPADEDecoderBlock(ch * 4, ch * 2, attr_channels=num_classes)
        # 256→128, H/4→H/2
        self.dec3 = SPADEDecoderBlock(ch * 2, ch, attr_channels=num_classes)
        # 128→64, H/2→H

        # ---- Output heads ----
        self.foreground_head = nn.Sequential(
            nn.Conv2d(ch, in_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )  # F_rgb ∈ [-1, 1]

        self.mask_head = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )  # M ∈ [0, 1]

        if encoder_stride == 4:
            self.enc4 = nn.Identity()
            self.bottleneck = nn.Sequential(*[ResBlock(ch*4, use_noise=True) for _ in range(num_res_blocks)])
            self.dec1 = nn.Identity()
        if scalar_noise:
            for module in self.modules():
                if isinstance(module, AdaptiveNoiseInjection):
                    module.weight = nn.Parameter(torch.zeros(1,1,1,1))
            # Paper injects independently learned scalar noise in encoder blocks too.
            for encoder in (self.enc1, self.enc2, self.enc3):
                channels = encoder[0].out_channels
                encoder.append(AdaptiveNoiseInjection(channels, scalar=True))

        # Apply weight initialisation
        self.apply(self._init_weights)

    # ----- weight init -----

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialise weights: He for ReLU layers, Xavier for output layers."""
        if isinstance(m, nn.Conv2d):
            # Detect output-head convolutions by their successors being
            # Tanh/Sigmoid — those are wrapped in nn.Sequential so we
            # cannot easily detect here.  Default to He (Kaiming).
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ----- helpers -----

    def _build_attr_map(
        self,
        c: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """Expand a category vector into a spatial attribute map for SPADE.

        Args:
            c: Category vector [B, num_classes].
            spatial_size: Target (H, W) for the attribute map.

        Returns:
            Attribute map [B, num_classes, H, W] with the category vector
            broadcast across spatial dims.
        """
        # c: [B, num_classes] → [B, num_classes, 1, 1] → [B, num_classes, H, W]
        return c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *spatial_size)

    # ----- forward -----

    def forward(self, x, c=None, spatial_map=None, noise_mode=None, generator=None):
        token = _noise_settings.set((noise_mode, generator))
        try:
            return self._forward(x, c, spatial_map)
        finally:
            _noise_settings.reset(token)

    def _forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        spatial_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate a defect (or restored) image via layer-wise composition.

        Args:
            x: Input RGB image tensor [B, 3, H, W], values in [-1, 1].
            c: Category vector [B, num_classes].  If ``None``, defaults to
               an all-ones vector (single defect class).
            spatial_map: Optional [B, num_classes, H, W] control in [0, 1],
                multiplied by the category vector to form the SPADE input.
                It is neither the generated alpha nor an export-time ROI.

        Returns:
            Dictionary with keys:
                ``'output'``     – composed RGB image [B, 3, H, W]
                ``'foreground'`` – raw foreground F [B, 3, H, W]
                ``'mask'``       – spatial distribution map M [B, 1, H, W]
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Generator input must be [B,3,H,W]")
        if any(n % self.encoder_stride for n in x.shape[2:]):
            raise ValueError("Image dimensions must be divisible by encoder stride")
        batch_size = x.size(0)
        input_rgb = x  # keep reference for composition

        # Default category: all-ones (single scratch class)
        if c is None:
            c = torch.ones(batch_size, self.num_classes, device=x.device)

        # ---- Encode ----
        e1 = self.enc1(x)    # [B,  64, H,   W  ]
        e2 = self.enc2(e1)   # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)   # [B, 256, H/4, W/4]
        e4 = self.enc4(e3)   # [B, 512, H/8, W/8]

        # ---- Bottleneck ----
        z = self.bottleneck(e4)  # [B, 512, H/8, W/8]

        # ---- Decode (SPADE conditioned on category) ----
        # Build attribute map at coarsest decoder resolution and let SPADE
        # blocks internally resize as needed.
        if c.shape != (batch_size, self.num_classes):
            raise ValueError("Category vector must be [B,num_classes]")
        if spatial_map is not None:
            expected = (batch_size, self.num_classes, *x.shape[2:])
            if spatial_map.shape != expected or not torch.isfinite(spatial_map).all():
                raise ValueError("Spatial map must be finite [B,num_classes,H,W]")
            if (spatial_map < 0).any() or (spatial_map > 1).any():
                raise ValueError("Spatial map values must lie in [0,1]")
            # Spatial control modulates the requested category, rather than
            # replacing it. In particular, the paper baseline's zero-valued
            # restoration category must remain zero even if a map was supplied.
            attr_map = spatial_map.to(device=x.device, dtype=x.dtype) * c.to(
                device=x.device, dtype=x.dtype
            ).unsqueeze(-1).unsqueeze(-1)
        else:
            attr_map = self._build_attr_map(c, z.shape[2:])

        d1 = z if self.encoder_stride == 4 else self.dec1(z, attr_map)     # [B, 256, H/4, W/4]
        d2 = self.dec2(d1, attr_map)    # [B, 128, H/2, W/2]
        d3 = self.dec3(d2, attr_map)    # [B,  64, H,   W  ]

        # ---- Output heads ----
        foreground = self.foreground_head(d3)  # [B, 3, H, W]
        mask = self.mask_head(d3)              # [B, 1, H, W]

        # ---- Layer-wise composition ----
        output = compose_layers(input_rgb, foreground, mask)

        return {
            "output": output,
            "foreground": foreground,
            "mask": mask,
        }
