"""Display measured environment facts without guessing batch size or 1024 VRAM fit."""
from __future__ import annotations

try:
    from scripts.gpu_diagnostics import main
except ModuleNotFoundError:
    from gpu_diagnostics import main


if __name__ == '__main__':
    raise SystemExit(main())
