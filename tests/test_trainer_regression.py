from pathlib import Path
import torch
from torch import nn
import pytest
from src.engine.trainer import DefectGANTrainer

class TinyModel(nn.Module):
    num_classes = 1
    def __init__(self):
        super().__init__()
        self.G_n2d = nn.Conv2d(3,3,1)
        self.G_d2n = nn.Conv2d(3,3,1)
        self.D_defect = nn.Conv2d(3,1,1)
        self.D_normal = nn.Conv2d(3,1,1)
    def defacement(self,x,c=None):
        y=self.G_n2d(x)
        return {'output':y,'mask':y[:,:1].sigmoid()}
    def restoration(self,x,c=None):
        y=self.G_d2n(x)
        return {'output':y,'mask':y[:,:1].sigmoid()}
    def discriminate_defect(self,x):
        return [(self.D_defect(x), x.mean((2,3))[:,:1])]
    def discriminate_normal(self,x):
        return [(self.D_normal(x), x.mean((2,3))[:,:1])]

class TinyLoss(nn.Module):
    lambda_rec=0
    def compute_discriminator_loss(self,**kw):
        loss=sum(v[0][0].square().mean() for k,v in kw.items() if k.startswith('d_'))
        return loss,{'d_loss':loss.item()}
    def compute_generator_loss(self,**kw):
        loss=kw['recon_normal']['output'].square().mean()+kw['recon_defect']['output'].square().mean()
        return loss,{'g_loss':loss.item(),'g_cyc_normal':loss.item(),'g_mask_defect':0.5}

def make_trainer(tmp_path,n=5,accum=4,n_critic=1):
    batch={'normal':torch.ones(1,3,4,4),'defect':torch.zeros(1,3,4,4)}
    cfg={'training':{'epochs':1,'mixed_precision':'none','gradient_accumulation_steps':accum,'n_critic':n_critic,'validation_interval':99,'checkpoint_interval':10}}
    return DefectGANTrainer(cfg,TinyModel(),TinyLoss(),[batch]*n,[],torch.device('cpu'),tmp_path)

@pytest.mark.parametrize('n,accum,expected',[(5,4,2),(2,4,1),(41,4,11)])
def test_accumulation_flushes_tail_and_changes_both_networks(tmp_path,n,accum,expected):
    trainer=make_trainer(tmp_path,n,accum)
    before=[p.detach().clone() for p in trainer.model.parameters()]
    trainer.train()
    assert trainer.generator_updates==expected
    assert trainer.discriminator_updates==expected
    assert all(p.grad is None for p in trainer.model.parameters())
    assert any(not torch.equal(a,b) for a,b in zip(before,trainer.model.parameters()))

def test_best_saved_independent_of_periodic_checkpoint(tmp_path):
    trainer=make_trainer(tmp_path,1,1)
    trainer.config['training']['validation_interval']=1
    trainer.val_loader=[{}]
    trainer.evaluator.evaluate=lambda *a,**k:{'val_cyc_normal':0.2,'val_cyc_defect':0.3,'val_mask_defect_mean':0.1}
    trainer.train()
    assert (trainer.checkpoint_dir/'best.pth').exists()
    assert (trainer.checkpoint_dir/'latest.pth').exists()

def test_safe_resume_and_unsafe_rejection(tmp_path):
    trainer=make_trainer(tmp_path,2,1)
    trainer.train()
    new=make_trainer(tmp_path/'new',2,1)
    new.load_checkpoint(trainer.checkpoint_dir/'latest.pth')
    assert new.current_epoch==1
    assert new.generator_updates==2
    trainer.save_checkpoint(tmp_path/'unsafe.pth',safe_to_resume=False)
    with pytest.raises(ValueError,match='unsafe|boundary'):
        new.load_checkpoint(tmp_path/'unsafe.pth')

def test_critic_schedule_counts_optimizer_groups(tmp_path):
    trainer=make_trainer(tmp_path,10,2,n_critic=5)
    trainer.train()
    assert trainer.discriminator_updates==5
    assert trainer.generator_updates==1

@pytest.mark.parametrize('mode',['oops','fp64'])
def test_invalid_amp_mode_rejected(tmp_path,mode):
    trainer=make_trainer(tmp_path)
    cfg={**trainer.config,'training':{**trainer.config['training'],'mixed_precision':mode}}
    with pytest.raises(ValueError,match='precision'):
        DefectGANTrainer(cfg,TinyModel(),TinyLoss(),[],[],torch.device('cpu'),tmp_path)

def test_tail_normalized_by_actual_microbatch_count(tmp_path):
    torch.manual_seed(7)
    tail=make_trainer(tmp_path/'tail',2,4)
    torch.manual_seed(7)
    full=make_trainer(tmp_path/'full',2,2)
    tail.train()
    full.train()
    for a,b in zip(tail.model.parameters(),full.model.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for left,right in zip(tail.opt_d.state.values(),full.opt_d.state.values()):
        torch.testing.assert_close(left['exp_avg'],right['exp_avg'],rtol=0,atol=0)
    for left,right in zip(tail.opt_g.state.values(),full.opt_g.state.values()):
        torch.testing.assert_close(left['exp_avg'],right['exp_avg'],rtol=0,atol=0)


def test_no_validation_does_not_create_best(tmp_path):
    trainer=make_trainer(tmp_path,1,1)
    trainer.train()
    assert not (trainer.checkpoint_dir/'best.pth').exists()


def test_unsafe_emergency_does_not_replace_latest(tmp_path):
    trainer=make_trainer(tmp_path,1,1)
    trainer.train()
    original=(trainer.checkpoint_dir/'latest.pth').read_bytes()
    trainer.save_checkpoint(trainer.checkpoint_dir/'unsafe.pth',safe_to_resume=False)
    assert (trainer.checkpoint_dir/'latest.pth').read_bytes()==original


def test_safe_resume_next_update_matches_uninterrupted(tmp_path):
    import random
    import numpy as np
    trainer=make_trainer(tmp_path/'one',2,1)
    trainer.train()
    checkpoint=tmp_path/'resume_original.pth'
    checkpoint.write_bytes((trainer.checkpoint_dir/'latest.pth').read_bytes())
    expected_rng=(random.random(),np.random.rand(),torch.rand(2))
    trainer.epochs=2
    trainer.train()
    new=make_trainer(tmp_path/'two',2,1)
    new.load_checkpoint(checkpoint)
    actual_rng=(random.random(),np.random.rand(),torch.rand(2))
    assert expected_rng[:2]==actual_rng[:2]
    torch.testing.assert_close(expected_rng[2],actual_rng[2])
    new.epochs=2
    new.train()
    assert new.generator_updates==trainer.generator_updates==4
    for a,b in zip(trainer.model.parameters(),new.model.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_paper_full_objective_update_resume_and_diagnostics(tmp_path):
    from src.models.defect_gan import DefectGAN
    from src.losses.gan_losses import PaperDefectGANLoss
    from src.engine.evaluator import DefectGANEvaluator
    torch.set_num_threads(1)
    def create(directory):
        model=DefectGAN(base_channels_g=4,base_channels_d=4,num_res_blocks=1,num_disc_scales=1,
                        architecture='defectgan_paper_baseline',num_disc_layers=3)
        cfg={'training':{'epochs':1,'mixed_precision':'none','gradient_accumulation_steps':1,'n_critic':1,
                         'scheduler':{'type':'linear_decay','decay_start_epoch':1}}}
        batch={'normal':torch.rand(1,3,32,32)*2-1,'defect':torch.rand(1,3,32,32)*2-1}
        return DefectGANTrainer(cfg,model,PaperDefectGANLoss(),[batch],[batch],torch.device('cpu'),directory)
    trainer=create(tmp_path/'first')
    before_g=[p.detach().clone() for p in trainer.g_params]
    before_d=[p.detach().clone() for p in trainer.d_params]
    trainer.train(dry_run=True)
    assert any(not torch.equal(a,b) for a,b in zip(before_g,trainer.g_params))
    assert any(not torch.equal(a,b) for a,b in zip(before_d,trainer.d_params))
    restored=create(tmp_path/'second')
    restored.load_checkpoint(trainer.checkpoint_dir/'latest.pth')
    assert restored.sched_g.state_dict()==trainer.sched_g.state_dict()
    restored.train(dry_run=True)
    assert restored.generator_updates==2
    assert restored.discriminator_updates==2
    assert (restored.output_dir/'validation'/'epoch_0001.json').exists()
    assert (restored.output_dir/'dry_run_evidence.json').exists()
    assert restored.model.training


def test_partial_dry_run_checkpoint_cannot_resume_or_replace_latest(tmp_path):
    trainer=make_trainer(tmp_path,5,2)
    trainer.train(dry_run=True)
    checkpoint=trainer.checkpoint_dir/'dry_run_partial_unsafe.pth'
    assert checkpoint.exists()
    assert not (trainer.checkpoint_dir/'latest.pth').exists()
    with pytest.raises(ValueError,match='unsafe|boundary'):
        make_trainer(tmp_path/'other').load_checkpoint(checkpoint)

def test_direct_microstep_checkpoint_is_not_resumable(tmp_path):
    trainer=make_trainer(tmp_path,1,1)
    batch=trainer.train_loader[0]
    trainer.train_step(batch['normal'],batch['defect'],0)
    trainer.save_checkpoint(tmp_path/'mid_epoch.pth')
    with pytest.raises(ValueError,match='unsafe'):
        trainer.load_checkpoint(tmp_path/'mid_epoch.pth')


def test_precision_policy_on_cpu_and_mock_cuda(tmp_path,monkeypatch):
    from src.engine.trainer import resolve_precision
    assert resolve_precision('auto',torch.device('cpu'))==(False,torch.float32,False)
    for mode in ('fp16','bf16'):
        with pytest.raises(ValueError,match='CUDA'):
            resolve_precision(mode,torch.device('cpu'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    assert resolve_precision('bf16',torch.device('cuda'))==(True,torch.bfloat16,False)
    assert resolve_precision('fp16',torch.device('cuda'))==(True,torch.float16,True)
    assert resolve_precision('auto',torch.device('cuda'))==(True,torch.bfloat16,False)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    with pytest.raises(ValueError,match='unsupported'):
        resolve_precision('bf16',torch.device('cuda'))
    assert resolve_precision('auto',torch.device('cuda'))==(True,torch.float16,True)

def test_generator_group_uses_completed_critic_update(tmp_path):
    trainer=make_trainer(tmp_path,4,2,n_critic=2)
    original=trainer._generator_step
    versions=[]
    def observe(*args,**kwargs):
        versions.append((trainer.discriminator_updates,[p.detach().clone() for p in trainer.d_params]))
        return original(*args,**kwargs)
    trainer._generator_step=observe
    trainer.train()
    assert [v[0] for v in versions]==[2,2]
    for a,b in zip(versions[0][1],versions[1][1]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_generator_group_uses_same_critic_weights_for_each_microbatch(tmp_path):
    trainer=make_trainer(tmp_path,2,2)
    observed=[]
    original=trainer.model.discriminate_defect
    def capture(image):
        if image.requires_grad:
            observed.append(trainer.model.D_defect.weight.detach().clone())
        return original(image)
    trainer.model.discriminate_defect=capture
    trainer.train()
    assert len(observed)==2
    torch.testing.assert_close(observed[0],observed[1],rtol=0,atol=0)


def test_smoke_subset_checkpoint_rejected_by_baseline(tmp_path):
    smoke=make_trainer(tmp_path/'smoke',1,1)
    smoke.config['training']['smoke_subset_samples']=1
    smoke.train()
    baseline=make_trainer(tmp_path/'baseline',1,1)
    with pytest.raises(ValueError,match='smoke subset'):
        baseline.load_checkpoint(smoke.checkpoint_dir/'latest.pth')


def test_disabled_autocast_does_not_construct_torch_context(tmp_path,monkeypatch):
    trainer=make_trainer(tmp_path)
    def unsupported(*args,**kwargs):
        raise RuntimeError('disabled autocast must not be constructed')
    monkeypatch.setattr(torch,'autocast',unsupported)
    with trainer._autocast():
        pass
    with trainer._autocast(False):
        pass


class SimulatedGradScaler:
    """CPU-compatible mock replicating PyTorch GradScaler unscale/skip/backoff lifecycle."""
    def __init__(self, initial_scale=65536.0, growth_factor=2.0, backoff_factor=0.5):
        self._scale = float(initial_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._enabled = True
        self._has_inf = False

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale

    def scale(self, outputs):
        return outputs

    def unscale_(self, opt):
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    self._has_inf = True

    def step(self, opt):
        if not self._has_inf:
            opt.step()

    def update(self):
        if self._has_inf:
            self._scale *= self._backoff_factor
            self._has_inf = False

    def state_dict(self):
        return {'scale': self._scale}

    def load_state_dict(self, d):
        self._scale = d['scale']


def test_simulated_fp16_overflow_skips_and_updates_scale(tmp_path):
    """Simulated regression: FP16 overflow triggers GradScaler skip and scale reduction, not ValueError."""
    trainer = make_trainer(tmp_path, 2, 1)
    trainer.use_scaler = True
    trainer.scaler_d = SimulatedGradScaler(initial_scale=65536.0)
    batch = trainer.train_loader[0]
    # Compute normal train step to populate grads
    trainer.train_step(batch['normal'], batch['defect'], 0, group_size=1, group_end=False, update_generator=False)
    # Inject FP16 overflow into discriminator grads
    for p in trainer.d_params:
        if p.grad is not None:
            p.grad.fill_(float('inf'))
    before_params = [p.detach().clone() for p in trainer.d_params]
    norm = trainer._step_optimizer(generator=False)
    # Step should be skipped: no param changes, scale halved, counters tracked
    assert trainer.discriminator_skipped_steps == 1
    assert trainer.discriminator_updates == 0
    assert trainer.scaler_d.get_scale() == 32768.0
    for before, current in zip(before_params, trainer.d_params):
        torch.testing.assert_close(before, current)
    assert all(p.grad is None for p in trainer.d_params)


def test_simulated_5d_1g_schedule_with_fp16_overflow_skip(tmp_path):
    """5D:1G schedule policy: G updates only after 5 actual SUCCESSFUL D updates, skipping over overflow."""
    # 6 microbatches, accum=1, n_critic=5
    trainer = make_trainer(tmp_path, 6, 1, n_critic=5)
    trainer.use_scaler = True
    scaler_d = SimulatedGradScaler(initial_scale=65536.0)
    trainer.scaler_d = scaler_d
    original_step_opt = trainer._step_optimizer
    step_call_count = [0]

    def flaky_step_optimizer(generator):
        if not generator:
            step_call_count[0] += 1
            if step_call_count[0] == 3:
                # Inject overflow at the 3rd D step
                for p in trainer.d_params:
                    if p.grad is not None:
                        p.grad.fill_(float('inf'))
        return original_step_opt(generator)

    trainer._step_optimizer = flaky_step_optimizer
    trainer.train()
    # Total D attempts: 6. 1 skipped, 5 successful.
    assert trainer.discriminator_skipped_steps == 1
    assert trainer.discriminator_updates == 5
    # G ran exactly once when D achieved 5 successful updates (at microbatch 6, not 5)
    assert trainer.generator_updates == 1


def test_scaler_disabled_non_finite_gradients_raise_error(tmp_path):
    """When scaler is disabled (FP32/none), non-finite gradients must fail-closed with ValueError."""
    trainer = make_trainer(tmp_path, 1, 1)
    trainer.use_scaler = False
    assert not trainer.scaler_d.is_enabled()
    batch = trainer.train_loader[0]
    trainer.train_step(batch['normal'], batch['defect'], 0, group_size=1, group_end=False, update_generator=False)
    for p in trainer.d_params:
        if p.grad is not None:
            p.grad.fill_(float('nan'))
    with pytest.raises(ValueError, match="Non-finite gradients"):
        trainer._step_optimizer(generator=False)


def test_validation_failure_preserves_completed_epoch_checkpoint(tmp_path):
    """Validation failure/OOM must preserve the completed epoch's checkpoint and not write best.pth."""
    trainer = make_trainer(tmp_path, 1, 1)
    trainer.config['training']['validation_interval'] = 1
    trainer.val_loader = [{}]
    def fail_eval(*a, **k):
        raise RuntimeError("Simulated validation OOM or evaluation failure")
    trainer.evaluator.evaluate = fail_eval
    with pytest.raises(RuntimeError, match="Simulated validation OOM"):
        trainer.train()
    # Completed epoch checkpoint was safely persisted BEFORE validation
    latest = trainer.checkpoint_dir / 'latest.pth'
    assert latest.exists(), "latest.pth must be saved before validation execution"
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    assert ckpt['next_epoch'] == 1
    assert ckpt['safe_to_resume'] is True
    # best.pth must NOT exist since validation failed
    assert not (trainer.checkpoint_dir / 'best.pth').exists()


def test_evaluator_amp_policy_and_memory_reporting(tmp_path):
    """Evaluator adheres to specified AMP policy and records diagnostics."""
    from src.engine.evaluator import DefectGANEvaluator
    val_loader = [{'normal': torch.ones(1, 3, 16, 16), 'defect': torch.zeros(1, 3, 16, 16)}]
    evaluator = DefectGANEvaluator(TinyModel(), TinyLoss(), val_loader, torch.device('cpu'),
                                  tmp_path, use_amp=False, amp_dtype=torch.float32)
    metrics = evaluator.evaluate(0, save_samples=False)
    assert 'val_cyc_normal' in metrics
    eval_json = tmp_path / 'validation' / 'epoch_0000.json'
    assert eval_json.exists()
    import json
    data = json.loads(eval_json.read_text(encoding='utf-8'))
    assert data['amp_enabled'] is False
    assert data['amp_dtype'] == 'none'
    assert 'metrics' in data


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real CUDA device for hardware execution")
def test_real_cuda_amp_gradscaler_overflow(tmp_path):
    """Real CUDA test: verifies native torch.cuda.amp.GradScaler skips on hardware overflow."""
    device = torch.device('cuda')
    cfg = {'training': {'epochs': 1, 'mixed_precision': 'fp16', 'gradient_accumulation_steps': 1,
                        'n_critic': 1, 'validation_interval': 99, 'checkpoint_interval': 10}}
    batch = {'normal': torch.ones(1, 3, 16, 16, device=device), 'defect': torch.zeros(1, 3, 16, 16, device=device)}
    trainer = DefectGANTrainer(cfg, TinyModel(), TinyLoss(), [batch], [], device, tmp_path)
    assert trainer.use_scaler is True
    assert trainer.scaler_d.is_enabled()
    scale_start = trainer.scaler_d.get_scale()
    trainer.train_step(batch['normal'], batch['defect'], 0, group_size=1, group_end=False, update_generator=False)
    # Inject inf into CUDA tensor grad
    for p in trainer.d_params:
        if p.grad is not None:
            p.grad.fill_(float('inf'))
    trainer._step_optimizer(generator=False)
    assert trainer.discriminator_skipped_steps == 1
    assert trainer.discriminator_updates == 0
    assert trainer.scaler_d.get_scale() < scale_start


def test_peak_cuda_vram_train_val_overall_tracking(tmp_path, monkeypatch):
    """Regression test: evaluator reset does not erase training peak memory in dry_run_evidence.json."""
    from contextlib import nullcontext
    import json
    trainer = make_trainer(tmp_path, 1, 1)

    class MockCudaDevice:
        type = 'cuda'
        index = 0
        def __str__(self):
            return 'cuda:0'
        def __repr__(self):
            return 'MockCudaDevice(cuda:0)'

    mock_device = MockCudaDevice()
    trainer.device = mock_device
    trainer.evaluator.device = mock_device

    # Mock tensor movement to avoid PyTorch CPU-only "Torch not compiled with CUDA enabled"
    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *a, **kw: self)

    def mock_train_step(*a, **kw):
        trainer.discriminator_updates += 1
        for p in trainer.d_params:
            p.data.add_(1.0)
            break
        return {'d_loss': 0.1}

    def mock_generator_step(*a, **kw):
        trainer.generator_updates += 1
        for p in trainer.g_params:
            p.data.add_(1.0)
            break
        return {'g_loss': 0.1}

    monkeypatch.setattr(trainer, 'train_step', mock_train_step)
    monkeypatch.setattr(trainer, '_generator_step', mock_generator_step)

    current_state = {'reset_count': 0}

    def mock_reset(device=None):
        current_state['reset_count'] += 1

    def mock_max_allocated(device=None):
        if current_state['reset_count'] <= 1:
            return 500 * 1024 * 1024
        return 200 * 1024 * 1024

    def mock_max_reserved(device=None):
        if current_state['reset_count'] <= 1:
            return 600 * 1024 * 1024
        return 250 * 1024 * 1024

    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', mock_reset)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', mock_max_allocated)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', mock_max_reserved)

    def mock_evaluate(epoch, save_samples=True):
        mock_reset(mock_device)
        return {
            'val_cyc_normal': 0.1,
            'val_cyc_defect': 0.1,
            'val_peak_cuda_allocated_bytes': mock_max_allocated(mock_device),
            'val_peak_cuda_reserved_bytes': mock_max_reserved(mock_device),
        }

    monkeypatch.setattr(trainer.evaluator, 'evaluate', mock_evaluate)

    trainer.val_loader = [{'normal': torch.ones(1, 3, 4, 4), 'defect': torch.zeros(1, 3, 4, 4)}]
    trainer.train(dry_run=True)

    assert current_state['reset_count'] == 2

    evidence_file = trainer.output_dir / 'dry_run_evidence.json'
    assert evidence_file.exists()
    evidence = json.loads(evidence_file.read_text(encoding='utf-8'))

    # Verify train peak was preserved despite evaluator reset
    assert evidence['peak_cuda_train_allocated_bytes'] == 500 * 1024 * 1024
    assert evidence['peak_cuda_train_reserved_bytes'] == 600 * 1024 * 1024

    # Verify val peak was measured separately
    assert evidence['peak_cuda_val_allocated_bytes'] == 200 * 1024 * 1024
    assert evidence['peak_cuda_val_reserved_bytes'] == 250 * 1024 * 1024

    # Verify overall peak is max(train, val), not wiped out by evaluator reset
    assert evidence['peak_cuda_overall_allocated_bytes'] == 500 * 1024 * 1024
    assert evidence['peak_cuda_overall_reserved_bytes'] == 600 * 1024 * 1024
    assert evidence['peak_cuda_allocated_bytes'] == 500 * 1024 * 1024
    assert evidence['peak_cuda_reserved_bytes'] == 600 * 1024 * 1024


def test_evaluator_peak_cuda_vram_reporting(tmp_path, monkeypatch):
    """Evaluator resets CUDA peak stats on enter and records val peak stats in results."""
    from contextlib import nullcontext
    from src.engine.evaluator import DefectGANEvaluator

    class MockCudaDevice:
        type = 'cuda'
        index = 0

    mock_dev = MockCudaDevice()
    val_loader = [{'normal': torch.ones(1, 3, 4, 4), 'defect': torch.zeros(1, 3, 4, 4)}]
    evaluator = DefectGANEvaluator(TinyModel(), TinyLoss(), val_loader, mock_dev,
                                  tmp_path, use_amp=False, amp_dtype=torch.float32)

    reset_calls = []
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda dev: reset_calls.append(dev))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda dev: 123456)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda dev: 234567)

    # Avoid device mismatch in mock forward pass
    monkeypatch.setattr(evaluator, '_autocast', lambda: nullcontext())
    monkeypatch.setattr(evaluator.model, 'defacement', lambda x, c: {'output': x, 'mask': torch.zeros(1, 1, 4, 4)})
    monkeypatch.setattr(evaluator.model, 'restoration', lambda x, c: {'output': x, 'mask': torch.zeros(1, 1, 4, 4)})
    monkeypatch.setattr(evaluator.model, 'discriminate_defect', lambda x: [torch.zeros(1, 1, 4, 4)])
    monkeypatch.setattr(evaluator.model, 'discriminate_normal', lambda x: [torch.zeros(1, 1, 4, 4)])
    monkeypatch.setattr(evaluator.criterion, 'compute_generator_loss', lambda **kw: (torch.tensor(0.0), {'g_loss': 0.1}))
    monkeypatch.setattr(evaluator.criterion, 'compute_discriminator_loss', lambda **kw: (torch.tensor(0.0), {'d_loss': 0.1}))

    # Mock tensor .to() to bypass CUDA check on CPU torch
    orig_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *a, **kw: self)

    # Mock torch.ones to avoid device='cuda' error on CPU torch
    orig_ones = torch.ones
    monkeypatch.setattr(torch, 'ones', lambda *a, **kw: orig_ones(*a, **{k: v for k, v in kw.items() if k != 'device'}))

    metrics = evaluator.evaluate(0, save_samples=False)

    assert len(reset_calls) == 1
    assert metrics['val_peak_cuda_allocated_bytes'] == 123456
    assert metrics['val_peak_cuda_reserved_bytes'] == 234567
