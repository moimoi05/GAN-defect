"""Epoch monitoring and resumable validation-proxy stopping (never a realism score)."""
import json
import math
from pathlib import Path

class EarlyStopping:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = cfg.get('enabled', False)
        if cfg.get('monitor', 'val_reconstruction') != 'val_reconstruction' or cfg.get('mode', 'min') != 'min':
            raise ValueError('Only val_reconstruction in min mode is supported')
        self.patience = int(cfg.get('patience', 25))
        self.warmup = int(cfg.get('warmup_epochs', 30))
        self.delta = float(cfg.get('min_delta', .001))
        if self.patience < 1 or self.warmup < 0 or self.delta < 0:
            raise ValueError('Invalid early stopping configuration')
        self.best = float('inf'); self.best_epoch = None
        self.bad_epochs = 0; self.stopped = False

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('Non-finite validation reconstruction')
        improved = score < self.best - self.delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
        elif epoch > self.warmup:
            self.bad_epochs += 1
        self.stopped = self.enabled and epoch > self.warmup and self.bad_epochs >= self.patience
        return improved

    def state_dict(self):
        return {k: getattr(self, k) for k in ('best', 'best_epoch', 'bad_epochs', 'stopped', 'enabled', 'patience', 'warmup', 'delta')}

    def load_state_dict(self, state):
        for k in self.state_dict():
            if k in state: setattr(self, k, state[k])

class History:
    def __init__(self, output):
        self.path = Path(output) / 'history.jsonl'
        self.plots = Path(output) / 'plots'

    def write(self, row):
        rows = self.read()
        rows = [r for r in rows if r['epoch'] < row['epoch']] + [row]
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
        tmp.replace(self.path)

    def read(self):
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()] if self.path.exists() else []

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = self.read()
        self.plots.mkdir(exist_ok=True)
        groups = {
            'loss': ['g_loss','d_loss'],
            'components': sorted({k for r in rows for k in r if k.startswith(('g_', 'd_')) and k not in ('g_loss','d_loss') and 'norm' not in k}),
            'generator_components': sorted({k for r in rows for k in r if k.startswith('g_') and k not in ('g_loss','g_gradient_norm')}),
            'discriminator_components': sorted({k for r in rows for k in r if k.startswith('d_') and k not in ('d_loss','d_gradient_norm')}),
            'validation': ['val_reconstruction','val_rgb_change_mean','val_mask_area_fraction_defect','val_bg_change_outside_mask'],
            'gradient': ['g_gradient_norm','d_gradient_norm'],
            'vram': ['peak_allocated_gib','peak_reserved_gib'],
            'lr': ['lr_g','lr_d']
        }
        for name, keys in groups.items():
            fig, ax = plt.subplots(figsize=(8,4))
            for key in keys:
                points = [(r['epoch'], r[key]) for r in rows if r.get(key) is not None]
                if points: ax.plot(*zip(*points), label=key, marker='.')
            ax.set_xlabel('Epoch'); ax.set_title(name); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(self.plots / (name+'.png')); plt.close(fig)
