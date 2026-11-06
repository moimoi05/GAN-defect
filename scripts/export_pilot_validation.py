"""Re-evaluate every saved pilot checkpoint and export the full fixed validation set."""
import random
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets.f971_dataset import DefectGANDataset, verify_manifest_consistency
from src.models.defect_gan import DefectGAN
from src.losses.gan_losses import PaperDefectGANLoss
from src.engine.evaluator import DefectGANEvaluator

CONFIG=ROOT/'configs/defectgan_stability_pilot.yaml'
OUT=ROOT/'outputs/stability_pilot/defectgan_f971_stability_pilot_5ep_seed42'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8-sig'))
    dc=cfg['data']; mc=cfg['model']; tc=cfg['training']
    root=Path(dc.get('data_root','.'))
    manifest=verify_manifest_consistency(dc['val_manifest'],data_root=root)
    dataset=DefectGANDataset(manifest,data_root=root,split='validation',image_size=1024)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True)
    model=DefectGAN(in_channels=3,base_channels_g=mc['generator']['base_channels'],
      base_channels_d=mc['discriminator']['base_channels'],num_classes=mc['generator']['num_classes'],
      num_res_blocks=mc['generator']['num_resblocks'],num_disc_scales=mc['discriminator']['num_scales'],
      architecture=mc['architecture'],num_disc_layers=mc['discriminator']['num_layers']).cuda()
    criterion=PaperDefectGANLoss(**tc['loss_weights']).cuda()
    evaluator=DefectGANEvaluator(model,criterion,loader,torch.device('cuda'),OUT,use_amp=True,amp_dtype=torch.bfloat16)
    evaluator.save_all_samples=True
    for epoch in range(5):
        path=OUT/'checkpoints'/f'checkpoint_epoch_{epoch:04d}.pth'
        state=torch.load(path,map_location='cuda',weights_only=False)
        if state.get('training_scope')!='full_loader' or state.get('safe_to_resume') is not True:
            raise ValueError(f'Unsafe or non-pilot checkpoint: {path}')
        model.load_state_dict(state['model_state'])
        seed_all(42)
        metrics=evaluator.evaluate(epoch,save_samples=True)
        print(f'epoch={epoch+1} exported_pairs={len(dataset)} val_reconstruction={metrics["val_cyc_normal"]+metrics["val_cyc_defect"]:.6f}',flush=True)
        del state
    print(f'Exported 5 checkpoints x {len(dataset)} fixed validation pairs.',flush=True)

if __name__=='__main__': main()
