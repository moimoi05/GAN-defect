"""Measured GPU diagnostics; this does not estimate training fitness or run training."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def collect_diagnostics() -> dict:
    import torch
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                props = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                # Capability is reported; successful AMP training still needs GPU smoke.
                devices.append({
                    'index': index, 'name': props.name,
                    'compute_capability': [props.major, props.minor],
                    'total_vram_bytes': total, 'free_vram_bytes': free,
                    'fp16_hardware_supported': (props.major, props.minor) >= (5, 3),
                    'bf16_supported': bool(torch.cuda.is_bf16_supported()),
                })
    driver = {'status': 'NOT_AVAILABLE', 'stdout': '', 'stderr': '', 'returncode': None}
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            proc = subprocess.run([
                executable, '--query-gpu=name,driver_version,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ], capture_output=True, text=True, timeout=15)
            driver = {'status': 'PASS' if proc.returncode == 0 else 'FAIL',
                      'returncode': proc.returncode, 'stdout': proc.stdout.strip(),
                      'stderr': proc.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            driver = {**driver, 'status': 'FAIL', 'stderr': str(exc)}
    return {
        'python': sys.version, 'platform': platform.platform(),
        'machine': platform.machine(), 'torch': str(torch.__version__),
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'cuda_available': available, 'gpu_count': len(devices), 'devices': devices,
        'nvidia_smi': driver, 'gpu_training_tested': False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = collect_diagnostics()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
