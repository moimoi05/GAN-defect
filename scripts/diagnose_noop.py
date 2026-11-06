import sys,json,csv,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image,ImageDraw
from src.models.defect_gan import DefectGAN
from src.datasets.f971_dataset import DefectGANDataset
from src.losses.gan_losses import PaperDefectGANLoss
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'outputs/stability_pilot/defectgan_f971_stability_pilot_5ep_seed42'; OUT=ROOT/'reports/noop_diagnosis'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
ds=DefectGANDataset(ROOT/'training/manifests/val.csv',data_root=ROOT,split='validation')
m=DefectGAN(architecture='defectgan_paper_baseline').cuda(); crit=PaperDefectGANLoss()
rows=[]; grads=[]
def rgb(t): return Image.fromarray(((t.detach().float().cpu().squeeze().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8'))
for ep in range(5):
 state=torch.load(RUN/'checkpoints'/f'checkpoint_epoch_{ep:04d}.pth',map_location='cpu',weights_only=False); m.load_state_dict(state['model_state']); del state
 m.eval(); torch.cuda.reset_peak_memory_stats()
 canvas=Image.new('RGB',(7*224,9*244),'white'); draw=ImageDraw.Draw(canvas)
 for i in range(len(ds)):
  b=ds[i]; n=b['normal'][None].cuda(); d=b['defect'][None].cuda(); c=torch.ones(1,1,device='cuda')
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   fd=m.defacement(n,c); fn=m.restoration(d,c); rd=m.discriminate_defect(d); rn=m.discriminate_normal(n); df=m.discriminate_defect(fd['output']); nf=m.discriminate_normal(fn['output'])
  delta=(fd['output']-n).abs(); mask=fd['mask'].float(); edge=(torch.nn.functional.avg_pool2d(n.mean(1,keepdim=True),3,1,1)-n.mean(1,keepdim=True)).abs()>.025
  row=dict(epoch=ep+1,sample=i,normal=b['normal_path'],defect=b['defect_path'],rgb=float(delta.mean()),mask_mean=float(mask.mean()),mask_area=float((mask>=.1).float().mean()),mask_near_zero=float((mask<.01).float().mean()),foreground_distance=float((fd['foreground']-n).abs().mean()),edge_delta=float(delta.expand_as(n)[edge.expand_as(n)].mean()),nonedge_delta=float(delta[~edge.expand_as(n)].mean()))
  for name,p in [('real_ng',rd),('real_ok',rn),('fake_ng',df),('fake_ok',nf)]: row[name+'_score']=float(p[0][0].float().mean()); row[name+'_logit']=float(p[0][1].float().mean())
  rows.append(row)
  imgs=[rgb(n),rgb(d),rgb(fd['output']),Image.fromarray((mask.cpu().squeeze().numpy()*255).astype('uint8')).convert('RGB'),rgb(fd['foreground']),rgb((delta*4).clamp(0,2)-1),rgb(fn['output'])]
  for j,im in enumerate(imgs): canvas.paste(im.resize((224,224)),(j*224,i*244+20))
  draw.text((0,i*244),f'{i}: OK | real NG | fake NG | alpha | foreground | abs diff x4 | restored NG',fill='black')
 canvas.save(OUT/f'epoch_{ep+1}_all9.png')
 # fixed pair 0, training stochastic graph, BF16 matching pilot; no optimizer steps
 random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
 m.train()
 for mod in m.modules(): mod.activation_checkpointing=True
 for p in m.D_defect.parameters(): p.requires_grad_(False)
 b=ds[0]; n=b['normal'][None].cuda(); d=b['defect'][None].cuda()
 names,params=zip(*m.G_n2d.named_parameters())
 for term_name in ['adv','cls','rec','sd_cyc','sd_con']:
  torch.manual_seed(42); torch.cuda.manual_seed_all(42)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   fd=m.defacement(n,c); fn=m.restoration(d,c); an=m.restoration(fd['output'],c); ad=m.defacement(fn['output'],c); pd=m.discriminate_defect(fd['output']); pn=m.discriminate_normal(fn['output'])
   terms={'adv':-(pd[0][0].mean()+pn[0][0].mean())/2,'cls':5*(torch.nn.functional.binary_cross_entropy_with_logits(pd[0][1],c)+torch.nn.functional.binary_cross_entropy_with_logits(pn[0][1],0*c))/2,'rec':5*((an['output']-n).abs().mean()+(ad['output']-d).abs().mean())/2,'sd_cyc':5*((fd['mask']-an['mask']).abs().mean()+(fn['mask']-ad['mask']).abs().mean())/2,'sd_con':sum(z['mask'].abs().mean() for z in [fd,fn,an,ad])/2}
  t=terms[term_name]
  gg=torch.autograd.grad(t,(*params,fd['mask']),allow_unused=True)
  gp,gm=gg[:-1],gg[-1]
  stats={}
  for key,pred in [('all',lambda n:True),('mask',lambda n:'mask_head' in n),('foreground',lambda n:'foreground_head' in n),('spade',lambda n:'spade' in n)]:
   stats[key]=sum(float((v.float()**2).sum()) for n,v in zip(names,gp) if v is not None and pred(n))**.5
  grads.append(dict(epoch=ep+1,term=term_name,weighted_loss=float(t),**stats,alpha_gradient_mean=None if gm is None else float(gm.float().mean()),alpha_positive_fraction=None if gm is None else float((gm>0).float().mean())))
  del terms,t,fd,fn,an,ad,pd,pn,gg,gp,gm
  torch.cuda.empty_cache()
  print('gradient',ep+1,term_name,stats,flush=True)
 print('epoch',ep+1,'done, peak GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
 (OUT/'gradients.json').write_text(json.dumps(grads,indent=2)); (OUT/'samples.json').write_text(json.dumps(rows,indent=2))
 with (OUT/'samples.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 torch.cuda.empty_cache()
print('complete',flush=True)
