"""Validation diagnostics; reconstruction metrics do not measure scratch realism."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from src.utils import save_rgb_image


class DefectGANEvaluator:
    def __init__(self, model, criterion, val_loader, device, output_dir):
        self.model, self.criterion, self.val_loader = model, criterion, val_loader
        self.device, self.output_dir = device, Path(output_dir)
        self.samples_dir = self.output_dir/'samples'
        self.samples_dir.mkdir(parents=True,exist_ok=True)

    @torch.no_grad()
    def evaluate(self, epoch, save_samples=True):
        previous = self.model.training
        self.model.eval()
        totals, count = {}, 0
        try:
            for index,batch in enumerate(self.val_loader):
                normal, defect = batch['normal'].to(self.device), batch['defect'].to(self.device)
                c = torch.ones(normal.size(0),self.model.num_classes,device=self.device)
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
                metrics.update(val_cyc_normal=(recon_n['output']-normal).abs().mean().item(),
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
                if save_samples and index == 0:
                    directory=self.samples_dir/f'epoch_{epoch:04d}'
                    for i in range(min(n,4)):
                        images={'01_normal_input':normal[i],'02_fake_defect':fake_d['output'][i],
                                '03_internal_alpha_debug':fake_d['mask'][i], '04_recon_normal':recon_n['output'][i],
                                '05_real_defect_unpaired_reference':defect[i],'06_fake_normal':fake_n['output'][i],
                                '07_recon_defect':recon_d['output'][i],
                                '08_rgb_difference_debug':(fake_d['output'][i]-normal[i]).abs().clamp(0,2)-1}
                        for name,image in images.items():
                            save_rgb_image(image,directory/f'sample_{i}_{name}.png')
            if count == 0:
                raise ValueError('Validation loader is empty; no valid checkpoint selection metric')
            result={k:v/count for k,v in totals.items()}
            directory=self.output_dir/'validation'
            directory.mkdir(parents=True,exist_ok=True)
            (directory/f'epoch_{epoch:04d}.json').write_text(json.dumps({
                'metrics':result,'sample_count':count,'critic_loss_excludes_gradient_penalty':True,
                'selection_metric':'val_cyc_normal + val_cyc_defect (reconstruction proxy, not realism)'},indent=2),encoding='utf-8')
            return result
        finally:
            self.model.train(previous)
