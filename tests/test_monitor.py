import json
import pytest
from src.engine.monitor import EarlyStopping, History

def test_warmup_patience_resume():
    cfg=dict(enabled=True, patience=2, warmup_epochs=3, min_delta=.1)
    stop=EarlyStopping(cfg)
    assert stop.update(1.,1)
    for epoch in (2,3):
        assert not stop.update(.95,epoch)
        assert stop.bad_epochs==0
    stop.update(.95,4)
    resumed=EarlyStopping(cfg); resumed.load_state_dict(stop.state_dict())
    resumed.update(.95,5)
    assert resumed.stopped and resumed.best_epoch==1

def test_disabled_and_invalid():
    stop=EarlyStopping(dict(patience=1,warmup_epochs=0))
    stop.update(1,1);stop.update(2,2)
    assert not stop.stopped
    with pytest.raises(ValueError): stop.update(float('nan'),3)
    with pytest.raises(ValueError): EarlyStopping(dict(monitor='g_loss'))

def test_history_resume_replaces_future(tmp_path):
    h=History(tmp_path)
    for epoch in (1,2,3): h.write(dict(epoch=epoch,g_loss=epoch))
    h.write(dict(epoch=2,g_loss=9))
    assert h.read()==[dict(epoch=1,g_loss=1),dict(epoch=2,g_loss=9)]
    h.plot()
    assert len(list(h.plots.glob('*.png')))==6

def test_checkpoint_gradient_equivalence():
    import copy
    import torch
    from src.models.generator import DefectGANGenerator
    from src.models.discriminator import StarGANDiscriminator
    for module, x in [(DefectGANGenerator(base_channels=4,num_res_blocks=1),torch.randn(1,3,32,32)),(StarGANDiscriminator(base_channels=4,num_layers=2),torch.randn(1,3,32,32))]:
        other=copy.deepcopy(module); other.activation_checkpointing=True
        gradients=[]
        for model in (module,other):
            torch.manual_seed(123)
            y=model(x)
            loss=y['output'].square().mean() if isinstance(y,dict) else sum(a.square().mean()+b.square().mean() for a,b in y)
            loss.backward()
            gradients.append([p.grad.clone() if p.grad is not None else None for p in model.parameters()])
        for a,b in zip(*gradients):
            if a is not None: torch.testing.assert_close(a,b)

def test_trainer_early_stop_checkpoint_resume(tmp_path):
    import torch
    from test_trainer_regression import make_trainer
    from src.engine.monitor import EarlyStopping
    t=make_trainer(tmp_path,5,1,n_critic=5)
    t.epochs=5; t.config['training']['validation_interval']=1
    t.early_stopping=EarlyStopping(dict(enabled=True,patience=1,warmup_epochs=0))
    t.val_loader=[{}]
    t.evaluator.evaluate=lambda *a,**k: dict(val_cyc_normal=.1,val_cyc_defect=.2)
    t.train()
    assert t.current_epoch==2 and t.early_stopping.stopped
    new=make_trainer(tmp_path/'resume',5,1,5)
    new.load_checkpoint(t.checkpoint_dir/'latest.pth')
    assert new.early_stopping.state_dict()==t.early_stopping.state_dict()
    assert new.generator_updates==2 and new.discriminator_updates==10
    assert len(t.history.read())==2

def test_progress_tracks_microbatches(tmp_path, monkeypatch):
    from test_trainer_regression import make_trainer
    import src.engine.trainer as training
    calls=[]
    class Bar:
        def __init__(self, **kw): calls.append(('total',kw['total']))
        def update(self, n): calls.append(('update',n))
        def set_postfix(self, values): calls.append(('postfix',values))
        def close(self): calls.append(('close',0))
    monkeypatch.setattr(training,'tqdm',Bar)
    t=make_trainer(tmp_path,5,2)
    t.train()
    assert sum(n for k,n in calls if k=='update')==5
    assert ('total',5) in calls and ('close',0) in calls
    assert len(t.history.read())==1
