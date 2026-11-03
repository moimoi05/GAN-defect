"""Training with optimizer-group accumulation and safe epoch-boundary checkpoints."""
from __future__ import annotations
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from src.engine.evaluator import DefectGANEvaluator

logger = logging.getLogger(__name__)


def unique_parameters(*modules):
    return list({id(p): p for m in modules for p in m.parameters()}.values())


def resolve_precision(mode, device):
    if mode not in ('none', 'fp16', 'bf16', 'auto'):
        raise ValueError(f'Invalid mixed_precision: {mode}')
    if device.type != 'cuda':
        if mode in ('fp16', 'bf16'):
            raise ValueError('Explicit CUDA mixed precision requires a CUDA device; use none on CPU')
        return False, torch.float32, False
    if not torch.cuda.is_available():
        raise ValueError('CUDA device requested but CUDA is unavailable')
    if mode == 'none':
        return False, torch.float32, False
    bf16 = torch.cuda.is_bf16_supported()
    if mode == 'bf16' and not bf16:
        raise ValueError('BF16 precision is unsupported by this GPU')
    dtype = torch.bfloat16 if mode == 'bf16' or (mode == 'auto' and bf16) else torch.float16
    return True, dtype, dtype == torch.float16


class DefectGANTrainer:
    def __init__(self, config, model, criterion, train_loader, val_loader, device, output_dir):
        self.config, self.device = config, device
        self.model, self.criterion = model.to(device), criterion.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg = config.get('training', {})
        self.use_amp, self.amp_dtype, self.use_scaler = resolve_precision(cfg.get('mixed_precision', 'auto'), device)
        self.evaluator = DefectGANEvaluator(model, criterion, val_loader, device, self.output_dir,
                                            use_amp=self.use_amp, amp_dtype=self.amp_dtype)
        opt = cfg.get('optimizer', {})
        self.g_params = unique_parameters(model.G_n2d, model.G_d2n)
        self.d_params = unique_parameters(model.D_defect, model.D_normal)
        common = dict(betas=(opt.get('beta1', 0.0), opt.get('beta2', .999)), weight_decay=opt.get('weight_decay', 0.0))
        self.opt_g = torch.optim.Adam(self.g_params, lr=opt.get('lr_g', .0001), **common)
        self.opt_d = torch.optim.Adam(self.d_params, lr=opt.get('lr_d', .0001), **common)
        self.grad_accum_steps = int(cfg.get('gradient_accumulation_steps', 1))
        self.n_critic = int(cfg.get('n_critic', 5 if getattr(criterion, 'paper_baseline', False) else 1))
        if min(self.grad_accum_steps, self.n_critic) < 1:
            raise ValueError('accumulation steps and n_critic must be positive')
        # torch.cuda.amp is available in the selected Docker PyTorch 2.1.2.
        self.scaler_g = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.epochs = int(cfg.get('epochs', 100))
        scfg = cfg.get('scheduler', {})
        decay = scfg.get('decay_start_epoch', self.epochs // 2)
        def factor(epoch):
            return 1.0 if epoch < decay else max(0., 1. - (epoch-decay)/max(1, self.epochs-decay))
        self.sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, factor) if scfg.get('type') == 'linear_decay' else None
        self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, factor) if self.sched_g else None
        self.current_epoch = self.global_step = self.generator_updates = self.discriminator_updates = 0
        self.generator_skipped_steps = self.discriminator_skipped_steps = 0
        self.best_val_loss = float('inf')
        self.at_epoch_boundary = True
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)

    def _autocast(self, enabled=None):
        active = self.use_amp if enabled is None else enabled
        if not active:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)

    def _toggle(self, generator):
        for p in self.g_params:
            p.requires_grad_(generator)
        for p in self.d_params:
            p.requires_grad_(not generator)

    def _step_optimizer(self, generator):
        opt, scaler = (self.opt_g, self.scaler_g) if generator else (self.opt_d, self.scaler_d)
        params = self.g_params if generator else self.d_params
        scaler.unscale_(opt)
        if not any(p.grad is not None for p in params):
            raise RuntimeError('Optimizer has no gradients')
        has_non_finite = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params)
        if not scaler.is_enabled() and has_non_finite:
            raise ValueError('Non-finite gradients; optimizer update refused')
        clip = self.config.get('training', {}).get('gradient_clip_norm')
        norm = torch.nn.utils.clip_grad_norm_(params, float(clip) if clip else float('inf'))
        scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
        scaler.step(opt)
        scaler.update()
        scale_after = scaler.get_scale() if scaler.is_enabled() else 1.0
        opt.zero_grad(set_to_none=True)
        skipped = scaler.is_enabled() and (has_non_finite or scale_after < scale_before)
        if skipped:
            if generator:
                self.generator_skipped_steps += 1
            else:
                self.discriminator_skipped_steps += 1
            logger.warning(
                "FP16 overflow detected: %s optimizer step skipped by GradScaler "
                "(scale: %s -> %s, total skipped: %d)",
                "generator" if generator else "discriminator",
                scale_before,
                scale_after,
                self.generator_skipped_steps if generator else self.discriminator_skipped_steps,
            )
        else:
            if generator:
                self.generator_updates += 1
            else:
                self.discriminator_updates += 1
        return float(norm)

    def train_step(self, normal, defect, step_in_epoch, *, group_size=None, group_end=None, update_generator=True):
        """One microbatch; group_size includes an incomplete tail's actual size."""
        self.at_epoch_boundary = False
        divisor = group_size or self.grad_accum_steps
        boundary = (step_in_epoch + 1) % self.grad_accum_steps == 0 if group_end is None else group_end
        c = torch.ones(normal.size(0), self.model.num_classes, device=self.device)
        self._toggle(False)
        with self._autocast():
            with torch.no_grad():
                fake_d = self.model.defacement(normal, c)
                fake_n = self.model.restoration(defect, c)
            kwargs = dict(d_defect_real=self.model.discriminate_defect(defect),
                          d_defect_fake=self.model.discriminate_defect(fake_d['output'].detach()),
                          d_normal_real=self.model.discriminate_normal(normal),
                          d_normal_fake=self.model.discriminate_normal(fake_n['output'].detach()),
                          target_cls_defect=c, target_cls_normal=torch.zeros_like(c))
        if getattr(self.criterion, 'paper_baseline', False):
            with self._autocast(False):
                gp = .5 * (self.criterion.gradient_penalty(self.model.discriminate_defect, defect.float(), fake_d['output'].float()) +
                           self.criterion.gradient_penalty(self.model.discriminate_normal, normal.float(), fake_n['output'].float()))
            kwargs['gradient_penalty'] = gp
        with self._autocast():
            d_loss, logs = self.criterion.compute_discriminator_loss(**kwargs)
        if not torch.isfinite(d_loss):
            raise ValueError('Non-finite discriminator loss')
        self.scaler_d.scale(d_loss/divisor).backward()
        if boundary:
            logs['d_gradient_norm'] = self._step_optimizer(False)
        if not update_generator:
            return logs
        glogs = self._generator_step(normal, defect, divisor, boundary)
        return {**logs, **glogs}

    def _generator_step(self, normal, defect, divisor, boundary):
        c = torch.ones(normal.size(0), self.model.num_classes, device=self.device)
        self._toggle(True)
        with self._autocast():
            fake_d = self.model.defacement(normal, c)
            fake_n = self.model.restoration(defect, c)
            recon_n = self.model.restoration(fake_d['output'], c)
            recon_d = self.model.defacement(fake_n['output'], c)
            identity = self.criterion.lambda_rec > 0 and not getattr(self.criterion, 'paper_baseline', False)
            g_loss, glogs = self.criterion.compute_generator_loss(
                normal_img=normal, defect_img=defect, fake_defect=fake_d, fake_normal=fake_n,
                recon_normal=recon_n, recon_defect=recon_d,
                id_defect=self.model.defacement(defect,c) if identity else None,
                id_normal=self.model.restoration(normal,c) if identity else None,
                d_defect_fake=self.model.discriminate_defect(fake_d['output']),
                d_normal_fake=self.model.discriminate_normal(fake_n['output']),
                target_cls_defect=c, target_cls_normal=torch.zeros_like(c))
        if not torch.isfinite(g_loss):
            raise ValueError('Non-finite generator loss')
        self.scaler_g.scale(g_loss/divisor).backward()
        if boundary:
            glogs['g_gradient_norm'] = self._step_optimizer(True)
        return glogs

    @staticmethod
    def _atomic_save(state, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    def save_checkpoint(self, path, is_best=False, *, safe_to_resume=None):
        safe = self.at_epoch_boundary if safe_to_resume is None else safe_to_resume
        state = dict(checkpoint_schema=2, epoch=self.current_epoch-1 if safe else self.current_epoch, next_epoch=self.current_epoch,
                     safe_to_resume=safe, resume_policy='epoch_boundary_only', global_step=self.global_step,
                     generator_updates=self.generator_updates, discriminator_updates=self.discriminator_updates,
                     generator_skipped_steps=self.generator_skipped_steps,
                     discriminator_skipped_steps=self.discriminator_skipped_steps,
                     scaler_g_scale=self.scaler_g.get_scale() if self.use_scaler else 1.0,
                     scaler_d_scale=self.scaler_d.get_scale() if self.use_scaler else 1.0,
                     model_state=self.model.state_dict(), opt_g_state=self.opt_g.state_dict(), opt_d_state=self.opt_d.state_dict(),
                     scaler_g_state=self.scaler_g.state_dict(), scaler_d_state=self.scaler_d.state_dict(),
                     sched_g_state=self.sched_g.state_dict() if self.sched_g else None,
                     sched_d_state=self.sched_d.state_dict() if self.sched_d else None,
                     config=self.config, best_val_loss=self.best_val_loss,
                     rng_state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                                'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                     architecture=getattr(self.model,'architecture','defectgan_custom_legacy'),
                     training_scope='smoke_subset' if self.config.get('smoke_subset') else 'full_loader')
        self._atomic_save(state,path)
        if safe:
            latest = self.checkpoint_dir/'latest.pth'
            if Path(path) != latest:
                self._atomic_save(state,latest)
            if is_best:
                self._atomic_save(state,self.checkpoint_dir/'best.pth')

    def load_checkpoint(self, checkpoint_path):
        # Training checkpoints contain Python/NumPy RNG state. Load trusted local checkpoints only.
        checkpoint = torch.load(checkpoint_path,map_location=self.device,weights_only=False)
        if checkpoint.get('checkpoint_schema') != 2 or not checkpoint.get('safe_to_resume'):
            raise ValueError('Checkpoint unsafe for resume: schema 2 epoch boundary required; use latest.pth')
        if checkpoint.get('architecture') != getattr(self.model,'architecture','defectgan_custom_legacy'):
            raise ValueError('Checkpoint architecture does not match configured model')
        saved_subset = checkpoint.get('config', {}).get('training', {}).get('smoke_subset_samples')
        current_subset = self.config.get('training', {}).get('smoke_subset_samples')
        if saved_subset != current_subset:
            raise ValueError('Checkpoint smoke subset does not match current training configuration')
        scope = 'smoke_subset' if self.config.get('smoke_subset') else 'full_loader'
        if checkpoint.get('training_scope', 'full_loader') != scope:
            raise ValueError('Smoke-subset checkpoints cannot resume full training')
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state'])
        for name in ('scaler_g','scaler_d','sched_g','sched_d'):
            obj, state = getattr(self,name), checkpoint.get(name+'_state')
            if obj is not None and state is not None:
                obj.load_state_dict(state)
        self.current_epoch = checkpoint['next_epoch']
        for name in ('global_step','generator_updates','discriminator_updates','best_val_loss'):
            setattr(self,name,checkpoint[name])
        self.generator_skipped_steps = checkpoint.get('generator_skipped_steps', 0)
        self.discriminator_skipped_steps = checkpoint.get('discriminator_skipped_steps', 0)
        rng = checkpoint['rng_state']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if torch.cuda.is_available() and rng['cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
        self.at_epoch_boundary = True

    def train(self, dry_run=False):
        cfg = self.config.get('training',{})
        for name in ('validation_interval','checkpoint_interval','sample_interval'):
            if int(cfg.get(name,1)) < 1:
                raise ValueError(f'{name} must be positive')
        total = len(self.train_loader)
        if total == 0:
            raise ValueError('Training loader is empty')
        limit = min(total, self.grad_accum_steps*self.n_critic) if dry_run else total
        if dry_run and (limit+self.grad_accum_steps-1)//self.grad_accum_steps < self.n_critic:
            raise ValueError('Dry-run loader too short for at least one generator update at configured n_critic')
        snapshots = [[p.detach().clone() for p in ps] for ps in (self.g_params,self.d_params)] if dry_run else None
        if dry_run and self.device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        end = self.current_epoch+1 if dry_run else self.epochs
        for epoch in range(self.current_epoch,end):
            self.current_epoch=epoch
            self.at_epoch_boundary=False
            self.model.train()
            self.opt_g.zero_grad(set_to_none=True)
            self.opt_d.zero_grad(set_to_none=True)
            try:
                iterator = iter(self.train_loader)
                processed = 0
                while processed < limit:
                    size = min(self.grad_accum_steps, limit - processed)
                    # Cache only one optimizer group on CPU, not activation graphs.
                    group = [next(iterator) for _ in range(size)]
                    for index, batch in enumerate(group):
                        logs = self.train_step(batch['normal'].to(self.device), batch['defect'].to(self.device),
                                               processed + index, group_size=size,
                                               group_end=index == size-1, update_generator=False)
                        self.global_step += 1
                        logger.info('epoch=%d microbatch=%d d_updates=%d d_losses=%s',
                                    epoch, processed+index, self.discriminator_updates, logs)
                    # 5D:1G schedule policy based on actual successful updates:
                    # G is updated only when the number of successful D updates satisfies:
                    # self.discriminator_updates >= (self.generator_updates + 1) * self.n_critic
                    # This guarantees:
                    # 1. Exact n_critic successful D updates precede each G update.
                    # 2. If D skips due to FP16 overflow, G waits until D achieves the required successful updates.
                    # 3. If G skips due to FP16 overflow, G retries on the next group once D succeeds, keeping pace.
                    # 4. G never updates before D has performed at least n_critic successful updates.
                    if self.discriminator_updates > 0 and self.discriminator_updates >= (self.generator_updates + 1) * self.n_critic:
                        for index, batch in enumerate(group):
                            glogs = self._generator_step(batch['normal'].to(self.device), batch['defect'].to(self.device),
                                                         size, index == size-1)
                            logger.info('epoch=%d microbatch=%d g_updates=%d g_losses=%s',
                                        epoch, processed+index, self.generator_updates, glogs)
                        logs = {**logs, **glogs}
                    processed += size
            except Exception:
                self.save_checkpoint(self.checkpoint_dir/f'emergency_unsafe_epoch_{epoch:04d}.pth',safe_to_resume=False)
                self.opt_g.zero_grad(set_to_none=True)
                self.opt_d.zero_grad(set_to_none=True)
                raise
            # A bounded run that did not exhaust its loader cannot resume at
            # the next epoch, even if all accumulated gradients were stepped.
            completed_epoch = limit == total
            self.at_epoch_boundary = completed_epoch
            if completed_epoch:
                for sched in (self.sched_g,self.sched_d):
                    if sched:
                        sched.step()
                # Advance epoch counter and safely persist completed epoch state first.
                # This guarantees validation failures or OOM cannot corrupt or erase trained weights.
                self.current_epoch = epoch + 1
                path = self.checkpoint_dir / 'latest.pth'
                if (epoch + 1) % cfg.get('checkpoint_interval', 10) == 0:
                    path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
                self.save_checkpoint(path, is_best=False)
            else:
                self.save_checkpoint(self.checkpoint_dir / 'dry_run_partial_unsafe.pth', safe_to_resume=False)

            is_best = False
            if self.val_loader and ((epoch + 1) % cfg.get('validation_interval', 1) == 0 or dry_run):
                try:
                    metrics = self.evaluator.evaluate(epoch, save_samples=(epoch + 1) % cfg.get('sample_interval', 5) == 0 or dry_run)
                    score = metrics['val_cyc_normal'] + metrics['val_cyc_defect']
                    if not np.isfinite(score):
                        raise ValueError(f'Non-finite validation metric: {score}')
                    is_best = score < self.best_val_loss
                    if is_best:
                        self.best_val_loss = score
                        # Only record best checkpoint after successful validation and improved metric
                        self.save_checkpoint(self.checkpoint_dir / 'best.pth', is_best=True)
                except Exception as exc:
                    logger.error(
                        "Validation failed at epoch %d (%s: %s). "
                        "Epoch training checkpoint is safely preserved at %s.",
                        epoch, type(exc).__name__, exc, path if completed_epoch else 'unsafe', exc_info=True
                    )
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise
        if dry_run:
            changed=[any(not torch.equal(a,b) for a,b in zip(before,params)) for before,params in zip(snapshots,(self.g_params,self.d_params))]
            evidence=dict(generator_changed=changed[0],discriminator_changed=changed[1],
                          generator_updates=self.generator_updates,discriminator_updates=self.discriminator_updates,
                          generator_skipped_steps=self.generator_skipped_steps,
                          discriminator_skipped_steps=self.discriminator_skipped_steps,
                          scaler_g_scale=self.scaler_g.get_scale() if self.use_scaler else 1.0,
                          scaler_d_scale=self.scaler_d.get_scale() if self.use_scaler else 1.0,
                          finite_losses=all(np.isfinite(v) for v in logs.values()),
                          checkpoint_schema=2,resume_policy='epoch_boundary_only',
                          completed_epoch=epoch if completed_epoch else None,next_epoch=self.current_epoch,
                          safe_to_resume=completed_epoch,
                          smoke_only=True,microbatches=limit,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None)
            (self.output_dir/'dry_run_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
            if not all(changed):
                raise RuntimeError('Dry-run did not change both generator and discriminator weights')
