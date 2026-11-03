"""Bounded CPU/GPU pipeline smoke. Every dependent stage fails closed.

CPU default is a small explicit adaptation for logic checks. --gpu-smoke selects
native 1024 paper config and requires CUDA; it never runs a full training session.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_CONTEXT = {}


def print_banner(message: str) -> None:
    print('\n' + '=' * 65 + '\n' + message + '\n' + '=' * 65, flush=True)


def resolve_output_dir(config: dict) -> Path:
    root = Path(os.environ.get('OUTPUT_DIR', config.get('output', {}).get('root', 'outputs')))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / '_smoke' / config.get('experiment', {}).get('name', 'defectgan_run')


def run_command(command: list[str]) -> bool:
    print('COMMAND: ' + subprocess.list2cmdline(command), flush=True)
    environment = dict(os.environ)
    environment.setdefault('OMP_NUM_THREADS', '1')
    environment.setdefault('MKL_NUM_THREADS', '1')
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    print(f'EXIT CODE: {result.returncode}', flush=True)
    return result.returncode == 0


def run_stage_1_dataset() -> bool:
    return run_command([sys.executable, str(PROJECT_ROOT / 'scripts/validate_dataset.py')])


def run_stage_2_components() -> bool:
    # The actual full-loss backward/update path is checked in stage 3. These
    # regressions independently exercise architecture, composition and gradients.
    test_args = ['tests/test_generator.py', 'tests/test_discriminator.py',
                 'tests/test_losses.py', 'tests/test_paper_baseline.py']
    if _CONTEXT.get('device') == 'cuda':
        # GPU acceptance procedure gate: verify native CUDA AMP GradScaler hardware execution
        test_args.extend(['tests/test_trainer_regression.py', '-k', 'test_real_cuda_amp_gradscaler_overflow'])
    return run_command([sys.executable, '-m', 'pytest', *test_args, '-q'])


def load_update_evidence(output_dir: Path) -> dict:
    evidence = json.loads((output_dir / 'dry_run_evidence.json').read_text(encoding='utf-8'))
    if not all(evidence.get(key) for key in ('generator_changed', 'discriminator_changed', 'finite_losses')):
        raise ValueError('dry-run did not prove finite losses and actual G/D weight updates')
    if any(evidence.get(key, 0) < 1 for key in ('generator_updates', 'discriminator_updates')):
        raise ValueError('dry-run optimizer update counters must both be positive')
    return evidence


def run_stage_3_trainer_dryrun() -> bool:
    output = _CONTEXT['output_dir']
    evidence_file = output / 'dry_run_evidence.json'
    # A stale successful smoke cannot substitute for evidence from this run.
    if evidence_file.exists():
        evidence_file.unlink()
    command = [sys.executable, '-m', 'src.train', '--config', str(_CONTEXT['config_path']),
               '--dry-run', '--device', _CONTEXT['device']]
    if not run_command(command):
        return False
    evidence = load_update_evidence(output)
    print(json.dumps(evidence, indent=2), flush=True)
    _CONTEXT['first_update_evidence'] = evidence
    return True


def run_stage_4_checkpoint_verify() -> bool:
    output = _CONTEXT['output_dir']
    checkpoint_path = output / 'checkpoints/latest.pth'
    # Trusted checkpoint created locally by the preceding stage.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = ('epoch', 'global_step', 'model_state', 'opt_g_state', 'opt_d_state', 'config')
    for key in required:
        if key not in checkpoint:
            raise ValueError(f'Missing checkpoint key: {key}')
    before = _CONTEXT['first_update_evidence']
    if not run_command([sys.executable, '-m', 'src.train', '--config', str(_CONTEXT['config_path']),
                        '--resume', str(checkpoint_path), '--dry-run', '--device', _CONTEXT['device']]):
        return False
    after = load_update_evidence(output)
    for key in ('generator_updates', 'discriminator_updates'):
        if after[key] <= before[key]:
            raise ValueError(f'Resume did not advance {key}')
    print('Checkpoint resume advanced G and D updates.', flush=True)
    return True


def run_stage_5_inference() -> bool:
    from PIL import Image
    output = _CONTEXT['output_dir'] / 'inference_smoke'
    output.mkdir(parents=True, exist_ok=True)
    if _CONTEXT['device'] == 'cuda':
        config = _CONTEXT['config']
        root = Path(os.environ.get('DATA_DIR', config['data'].get('data_root', '.')))
        manifest = Path(config['data']['train_manifest'])
        if not manifest.is_absolute():
            manifest = PROJECT_ROOT / manifest
        with manifest.open(encoding='utf-8-sig', newline='') as handle:
            sample = next(row for row in csv.DictReader(handle) if row['domain'] == 'clean')
        sample_input = root / Path(sample['patch_path'].replace('\\', '/'))
        patch_size = 1024
    else:
        # Synthetic image only proves RGB I/O mechanics; no learned quality claim.
        sample_input = output / 'synthetic_input.png'
        Image.new('RGB', (41, 35), (125, 130, 135)).save(sample_input)
        patch_size = 32
    sample_output = output / 'synthetic_output.png'
    if sample_output.exists():
        sample_output.unlink()
    command = [sys.executable, '-m', 'src.inference', '--checkpoint',
               str(_CONTEXT['output_dir'] / 'checkpoints/latest.pth'),
               '--input', str(sample_input), '--output', str(sample_output),
               '--mode', 'defacement', '--patch-size', str(patch_size),
               '--stride', str(patch_size), '--device', _CONTEXT['device'], '--seed', '42']
    if not run_command(command):
        return False
    with Image.open(sample_input) as source, Image.open(sample_output) as result:
        if result.mode != 'RGB' or result.size != source.size:
            raise ValueError('Inference must preserve dimensions and output RGB')
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--gpu-smoke', action='store_true')
    args = parser.parse_args(argv)
    config_path = args.config or PROJECT_ROOT / ('configs/defectgan_1024.yaml'
                                                if args.gpu_smoke else 'configs/smoke_test.yaml')
    try:
        config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
        device = 'cuda' if args.gpu_smoke else 'cpu'
        if args.gpu_smoke and not torch.cuda.is_available():
            raise ValueError('GPU smoke BLOCKED: CUDA unavailable')
        if args.gpu_smoke and config.get('data', {}).get('image_size') != 1024:
            raise ValueError('GPU smoke requires native 1024 configuration')
        _CONTEXT.update(config=config, config_path=config_path.resolve(), device=device,
                        output_dir=resolve_output_dir(config))
    except Exception as exc:
        print(f'FAILED setup: {exc}', flush=True)
        return 1
    stages = [run_stage_1_dataset, run_stage_2_components, run_stage_3_trainer_dryrun,
              run_stage_4_checkpoint_verify, run_stage_5_inference]
    results = {}
    failed = False
    for index, stage in enumerate(stages, 1):
        name = stage.__name__
        print_banner(f'STAGE {index}: {name}')
        if failed:
            results[name] = 'NOT_RUN (upstream stage failed)'
            continue
        try:
            passed = stage()
        except Exception as exc:
            print(f'FAILED {name}: {type(exc).__name__}: {exc}', flush=True)
            passed = False
        results[name] = 'PASS' if passed else 'FAIL'
        failed = not passed
    print_banner('BOUNDED SMOKE SUMMARY')
    print(json.dumps(results, indent=2), flush=True)
    print('SMOKE_FAILED' if failed else 'SMOKE_PASSED (mechanics only; model quality not evaluated)', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
