import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
CONFIG = ROOT / 'configs' / 'defectgan_stability_pilot.yaml'
OUT = ROOT / 'outputs' / 'stability_pilot' / 'defectgan_f971_stability_pilot_5ep_seed42'
LOG = ROOT / 'reports' / 'stability_pilot.log'
MAX_EPOCHS = 5
# A hard safety limit, selected at >3x the prior observed 76,784 G norm.
ABSOLUTE_GRAD_NORM_LIMIT = 250_000.0
RELATIVE_GRAD_NORM_LIMIT = 10.0
RELATIVE_MIN_NORM = 50_000.0


def emit(handle, message):
    print(message, flush=True)
    handle.write(message + '\n')
    handle.flush()


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['data']['image_size'] == 1024
    assert cfg['training']['batch_size'] == 1
    assert cfg['training']['gradient_accumulation_steps'] == 4
    assert cfg['training']['n_critic'] == 5
    assert cfg['training']['mixed_precision'] == 'bf16'
    assert cfg['training']['activation_checkpointing'] is True
    OUT.mkdir(parents=True, exist_ok=True)
    existing_rows = [json.loads(line) for line in (OUT / 'history.jsonl').read_text(encoding='utf-8').splitlines()] if (OUT / 'history.jsonl').exists() else []
    start_epoch = len(existing_rows) + 1
    previous_g = existing_rows[-1].get('g_gradient_norm') if existing_rows else None
    with LOG.open('w', encoding='utf-8') as log:
        emit(log, f'Pilot start: config={CONFIG} output={OUT}; max_epochs={MAX_EPOCHS}; fresh initialization; no resume checkpoint')
        emit(log, f'Stability guard: abs_norm={ABSOLUTE_GRAD_NORM_LIMIT:g}; relative={RELATIVE_GRAD_NORM_LIMIT:g}x above {RELATIVE_MIN_NORM:g}')
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            command = [str(PYTHON), '-u', '-m', 'src.train', '--config', str(CONFIG), '--device', 'cuda', '--max-epochs-this-run', '1']
            if epoch > 1:
                command.extend(['--resume', str(OUT / 'checkpoints' / 'latest.pth')])
            emit(log, f'\n=== PILOT EPOCH {epoch}/{MAX_EPOCHS} ===')
            started = time.perf_counter()
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
            if code:
                emit(log, f'TRAINING FAILED at epoch {epoch}, exit={code}; inspect emergency checkpoint and logs')
                return code
            history_path = OUT / 'history.jsonl'
            rows = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
            row = next((r for r in rows if r['epoch'] == epoch), None)
            if row is None:
                emit(log, f'ABORT: history missing epoch {epoch}')
                return 2
            expected = 72
            if row.get('discriminator_updates') != epoch * 18 or row.get('generator_updates') != (epoch * 18) // 5:
                emit(log, f'ABORT: unexpected 5D:1G counters at epoch {epoch}: G={row.get("generator_updates")} D={row.get("discriminator_updates")} (expected {(epoch*18)//5}/{epoch*18})')
                return 3
            gnorm = row.get('g_gradient_norm')
            dnorm = row.get('d_gradient_norm')
            values = [v for v in row.values() if isinstance(v, (int,float))]
            if not all(math.isfinite(v) for v in values):
                emit(log, f'ABORT: nonfinite history metric at epoch {epoch}')
                return 4
            if row.get('peak_allocated_gib', 99) >= 14.5 or row.get('peak_reserved_gib', 99) >= 15.5:
                emit(log, f'ABORT: VRAM guard at epoch {epoch}: allocated={row.get("peak_allocated_gib")} reserved={row.get("peak_reserved_gib")} GiB')
                return 5
            stop_reason = None
            if gnorm is not None and gnorm >= ABSOLUTE_GRAD_NORM_LIMIT:
                stop_reason = f'G gradient norm {gnorm:.4g} reached hard limit {ABSOLUTE_GRAD_NORM_LIMIT:g}'
            elif previous_g is not None and gnorm is not None and gnorm >= RELATIVE_MIN_NORM and gnorm / max(previous_g, 1e-12) >= RELATIVE_GRAD_NORM_LIMIT:
                stop_reason = f'G gradient norm jumped {gnorm / previous_g:.2f}x ({previous_g:.4g} -> {gnorm:.4g})'
            emit(log, f'Epoch {epoch} controller: G/D norm={gnorm}/{dnorm}; val_rec={row.get("val_reconstruction")}; bg_delta_outside_mask={row.get("val_bg_change_outside_mask")}; mask_area={row.get("val_mask_area_fraction_defect")}; VRAM={row.get("peak_allocated_gib")}/{row.get("peak_reserved_gib")} GiB; elapsed={row.get("epoch_seconds")}s')
            if stop_reason:
                evidence = {'status':'stopped_early','epoch':epoch,'reason':stop_reason,'history_row':row,'checkpoint':str(OUT/'checkpoints'/'latest.pth')}
                (OUT/'stability_guard.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
                emit(log, f'EARLY STOP: {stop_reason}; prior safe epoch checkpoint retained; evidence saved')
                break
            previous_g = gnorm
        else:
            (OUT/'stability_guard.json').write_text(json.dumps({'status':'completed_5_epochs','epochs':MAX_EPOCHS,'checkpoint':str(OUT/'checkpoints'/'latest.pth')},indent=2),encoding='utf-8')
        # Verify a checkpoint can restore all counters/state without taking another optimizer step.
        checkpoint = OUT / 'checkpoints' / 'latest.pth'
        verify = subprocess.run([str(PYTHON), '-u', '-m', 'src.train', '--config', str(CONFIG), '--device', 'cuda', '--resume', str(checkpoint), '--verify-resume-only'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        emit(log, '\n=== CHECKPOINT RESTORE VERIFICATION ===')
        for line in verify.stdout.splitlines(): emit(log, line)
        if verify.returncode:
            emit(log, f'CHECKPOINT RESTORE FAILED exit={verify.returncode}')
            return verify.returncode
        emit(log, f'Pilot complete. Review samples at {OUT / "samples"}; plots at {OUT / "plots"}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
