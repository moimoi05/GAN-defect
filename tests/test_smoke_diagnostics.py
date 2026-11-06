from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_smoke_test as smoke


def test_gpu_components_and_cuda_acceptance_run_separately(monkeypatch):
    commands = []
    monkeypatch.setitem(smoke._CONTEXT, 'device', 'cuda')
    monkeypatch.setattr(smoke, 'run_command', lambda command: commands.append(command) or True)
    assert smoke.run_stage_2_components()
    assert len(commands) == 2
    assert 'tests/test_paper_baseline.py' in commands[0]
    assert '-k' not in commands[0]
    assert 'tests/test_trainer_regression.py::test_real_cuda_amp_gradscaler_overflow' in commands[1]
    monkeypatch.setattr(smoke, 'run_command', lambda command: False)
    assert not smoke.run_stage_2_components()


def test_failed_stage_returns_nonzero_and_skips_dependencies(monkeypatch):
    called = []
    monkeypatch.setattr(smoke, 'run_stage_1_dataset', lambda: False)
    for name in ('run_stage_2_components', 'run_stage_3_trainer_dryrun',
                 'run_stage_4_checkpoint_verify', 'run_stage_5_inference'):
        monkeypatch.setattr(smoke, name, lambda: called.append('unexpected') or True)
    assert smoke.main([]) == 1
    assert not called


def test_stage_exception_is_failure(monkeypatch):
    def fail():
        raise ValueError('intentional regression failure')
    monkeypatch.setattr(smoke, 'run_stage_1_dataset', fail)
    assert smoke.main([]) == 1


def test_output_dir_environment_contract(monkeypatch, tmp_path):
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path))
    cfg = {'output': {'root': 'wrong'}, 'experiment': {'name': 'smoke'}}
    assert smoke.resolve_output_dir(cfg) == tmp_path / '_smoke' / 'smoke'


def test_checkpoint_rejects_no_optimizer_update(tmp_path):
    import json
    (tmp_path / 'dry_run_evidence.json').write_text(json.dumps({
        'generator_changed': False, 'discriminator_changed': True,
        'generator_updates': 0, 'discriminator_updates': 1, 'finite_losses': True,
    }))
    with pytest.raises(ValueError, match='update'):
        smoke.load_update_evidence(tmp_path)


def test_diagnostics_cpu_json_does_not_invent_gpu():
    from scripts.gpu_diagnostics import collect_diagnostics
    result = collect_diagnostics()
    assert isinstance(result['cuda_available'], bool)
    if not result['cuda_available']:
        assert result['gpu_count'] == 0
        assert result['devices'] == []
    assert 'recommendations' not in result


def test_intentional_failure_propagates_process_exit_code():
    code = (
        "import scripts.run_smoke_test as s; "
        "s.run_stage_1_dataset = lambda: False; "
        "raise SystemExit(s.main([]))"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'FAIL' in result.stdout
    assert 'NOT_RUN' in result.stdout


def test_successful_pipeline_returns_zero(monkeypatch):
    for name in ('run_stage_1_dataset', 'run_stage_2_components', 'run_stage_3_trainer_dryrun',
                 'run_stage_4_checkpoint_verify', 'run_stage_5_inference'):
        monkeypatch.setattr(smoke, name, lambda: True)
    assert smoke.main([]) == 0


def test_gpu_smoke_fails_closed_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(smoke.torch.cuda, 'is_available', lambda: False)
    assert smoke.main(['--gpu-smoke', '--config', 'configs/smoke_test.yaml']) == 1
    assert 'CUDA unavailable' in capsys.readouterr().out


def test_resume_requires_both_networks_to_advance(monkeypatch, tmp_path):
    from torch import save
    evidence = {'generator_changed': True, 'discriminator_changed': True,
                'generator_updates': 1, 'discriminator_updates': 5, 'finite_losses': True}
    import json
    (tmp_path / 'dry_run_evidence.json').write_text(json.dumps(evidence))
    (tmp_path / 'checkpoints').mkdir()
    save(dict.fromkeys(('epoch', 'global_step', 'model_state', 'opt_g_state', 'opt_d_state', 'config'), {}),
         tmp_path / 'checkpoints/latest.pth')
    monkeypatch.setattr(smoke, '_CONTEXT', {'output_dir': tmp_path,
        'config_path': Path('config.yaml'), 'device': 'cpu', 'first_update_evidence': evidence})
    monkeypatch.setattr(smoke, 'run_command', lambda command: True)
    with pytest.raises(ValueError, match='Resume did not advance'):
        smoke.run_stage_4_checkpoint_verify()


def test_stale_evidence_removed_before_failed_training(monkeypatch, tmp_path):
    stale = tmp_path / 'dry_run_evidence.json'
    stale.write_text('{}')
    monkeypatch.setattr(smoke, '_CONTEXT', {'output_dir': tmp_path,
        'config_path': Path('config.yaml'), 'device': 'cpu'})
    monkeypatch.setattr(smoke, 'run_command', lambda command: False)
    assert not smoke.run_stage_3_trainer_dryrun()
    assert not stale.exists()


def test_relative_output_is_resolved_against_subprocess_workdir(monkeypatch):
    monkeypatch.delenv('OUTPUT_DIR', raising=False)
    assert smoke.resolve_output_dir({'output': {'root': 'outputs/smoke'},
        'experiment': {'name': 'run'}}) == smoke.PROJECT_ROOT / 'outputs/smoke/_smoke/run'


def test_gpu_smoke_default_config_exists_and_is_usable(monkeypatch):
    monkeypatch.setattr(smoke.torch.cuda, 'is_available', lambda: True)
    for name in ('run_stage_1_dataset', 'run_stage_2_components', 'run_stage_3_trainer_dryrun',
                 'run_stage_4_checkpoint_verify', 'run_stage_5_inference'):
        monkeypatch.setattr(smoke, name, lambda: True)
    assert smoke.main(['--gpu-smoke']) == 0


def test_dry_run_dataset_is_bounded_complete_and_marked():
    from src.train import bounded_smoke_dataset
    from torch.utils.data import TensorDataset
    import torch
    dataset = TensorDataset(torch.arange(41))
    config = {'training': {'gradient_accumulation_steps': 4, 'n_critic': 5}}
    subset = bounded_smoke_dataset(dataset, config, batch_size=1)
    assert len(subset) == 20
    assert config['training']['smoke_subset_samples'] == 20
    assert len(dataset) == 41


def test_smoke_output_is_isolated_from_training(monkeypatch, tmp_path):
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path))
    assert smoke.resolve_output_dir({'experiment': {'name': 'baseline'}}) == tmp_path / '_smoke' / 'baseline'
