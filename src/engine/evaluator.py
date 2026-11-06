"""Validation diagnostics; reconstruction metrics do not measure scratch realism."""
from __future__ import annotations
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from src.utils import save_rgb_image


class DefectGANEvaluator:
    def __init__(self, model, criterion, val_loader, device, output_dir, use_amp=False, amp_dtype=torch.float16):
        self.model, self.criterion, self.val_loader = model, criterion, val_loader
        self.device, self.output_dir = device, Path(output_dir)
        self.use_amp, self.amp_dtype = use_amp, amp_dtype
        self.save_all_samples = False
        self.samples_dir = self.output_dir/'samples'
        self.samples_dir.mkdir(parents=True,exist_ok=True)

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)

    @torch.no_grad()
    def evaluate(self, epoch, save_samples=True):
        previous = self.model.training
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        totals, count = {}, 0
        try:
            for index,batch in enumerate(self.val_loader):
                normal, defect = batch['normal'].to(self.device), batch['defect'].to(self.device)
                c = torch.ones(normal.size(0),self.model.num_classes,device=self.device)
                with self._autocast():
                    fake_d = self.model.defacement(normal,c)
                    fake_n = self.model.restoration(defect,c)
                    recon_n = self.model.restoration(fake_d['output'],c)
                    recon_d = self.model.defacement(fake_n['output'],c)
                    d_fake_d = self.model.discriminate_defect(fake_d['output'])
                    d_fake_n = self.model.discriminate_normal(fake_n['output'])
                    identity = self.criterion.lambda_rec > 0 and not getattr(self.criterion,'paper_baseline',False)
                    _,glogs = self.criterion.compute_generator_loss(
                        normal_img=normal,defect_img=defect,fake_defect=fake_d,fake_normal=fake_n,
                        recon_normal=recon_n,recon_defect=recon_d,
                        id_defect=self.model.defacement(defect,c) if identity else None,
                        id_normal=self.model.restoration(normal,c) if identity else None,
                        d_defect_fake=d_fake_d,d_normal_fake=d_fake_n,
                        target_cls_defect=c,target_cls_normal=torch.zeros_like(c))
                    _,dlogs = self.criterion.compute_discriminator_loss(
                        d_defect_real=self.model.discriminate_defect(defect),d_defect_fake=d_fake_d,
                        d_normal_real=self.model.discriminate_normal(normal),d_normal_fake=d_fake_n,
                        target_cls_defect=c,target_cls_normal=torch.zeros_like(c))
                metrics = {'val_'+k:v for k,v in {**glogs,**dlogs}.items()}
                metrics['val_d_loss_without_gp'] = metrics.pop('val_d_loss')
                outside = (fake_d['mask'] < 0.1).to(fake_d['output'].dtype)
                outside_count = outside.sum().clamp_min(1.0) * 3
                background_delta = ((fake_d['output'] - normal).abs() * outside).sum() / outside_count
                metrics.update(val_bg_change_outside_mask=background_delta.item(),
                               val_mask_area_fraction_defect=(fake_d['mask'] >= 0.1).float().mean().item(),
                               val_cyc_normal=(recon_n['output']-normal).abs().mean().item(),
                               val_cyc_defect=(recon_d['output']-defect).abs().mean().item(),
                               val_mask_defect_mean=fake_d['mask'].mean().item(),
                               val_mask_normal_mean=fake_n['mask'].mean().item(),
                               val_mask_defect_std=fake_d['mask'].std(unbiased=False).item(),
                               val_rgb_change_mean=(fake_d['output']-normal).abs().mean().item())
                n = normal.size(0)
                for key,value in metrics.items():
                    if not torch.isfinite(torch.tensor(value)):
                        raise ValueError(f'Non-finite validation metric: {key}')
                    totals[key]=totals.get(key,0.)+value*n
                count+=n
                if save_samples and (index == 0 or self.save_all_samples):
                    directory=self.samples_dir/f'epoch_{epoch:04d}'
                    for i in range(min(n,4)):
                        sample_index = index * n + i
                        images={'01_normal_input':normal[i],'02_fake_defect':fake_d['output'][i],
                                '03_internal_alpha_debug':fake_d['mask'][i], '04_recon_normal':recon_n['output'][i],
                                '05_real_defect_unpaired_reference':defect[i],'06_fake_normal':fake_n['output'][i],
                                '07_recon_defect':recon_d['output'][i],
                                '08_rgb_difference_debug':(fake_d['output'][i]-normal[i]).abs().clamp(0,2)-1}
                        for name,image in images.items():
                            save_rgb_image(image,directory/f'sample_{sample_index}_{name}.png')
            if count == 0:
                raise ValueError('Validation loader is empty; no valid checkpoint selection metric')
            result={k:v/count for k,v in totals.items()}
            peak_alloc = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None
            peak_res = torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else None
            result['val_peak_cuda_allocated_bytes'] = peak_alloc
            result['val_peak_cuda_reserved_bytes'] = peak_res
            directory=self.output_dir/'validation'
            directory.mkdir(parents=True,exist_ok=True)
            meta = {
                'metrics':result,
                'sample_count':count,
                'critic_loss_excludes_gradient_penalty':True,
                'selection_metric':'val_cyc_normal + val_cyc_defect (reconstruction proxy, not realism)',
                'amp_enabled':self.use_amp,
                'amp_dtype':str(self.amp_dtype) if self.use_amp else 'none',
                'peak_cuda_allocated_bytes':peak_alloc,
                'peak_cuda_reserved_bytes':peak_res,
                'val_peak_cuda_allocated_bytes':peak_alloc,
                'val_peak_cuda_reserved_bytes':peak_res,
            }
            (directory/f'epoch_{epoch:04d}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            return result
        finally:
            self.model.train(previous)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
