"""Numerical contracts for paper baseline, using reduced CPU tensors only."""
import pytest
import torch

from src.models.defect_gan import DefectGAN
from src.models.generator import AdaptiveNoiseInjection



@pytest.mark.parametrize('alpha', [0., .3, 1.])
def test_composition_endpoints_and_gradient(alpha):
    from src.models.generator import compose_layers
    x = torch.full((1, 3, 4, 4), -.5, requires_grad=True)
    fg = torch.full_like(x, .7, requires_grad=True)
    mask = torch.full((1, 1, 4, 4), alpha, requires_grad=True)
    out = compose_layers(x, fg, mask)
    assert torch.allclose(out, x * (1-alpha) + fg * alpha)
    out.sum().backward()
    assert torch.allclose(fg.grad, torch.full_like(fg, alpha))
    assert torch.allclose(mask.grad, torch.full_like(mask, 3*1.2))


def test_eval_noise_is_seeded_and_independent_of_training():
    layer = AdaptiveNoiseInjection(3).eval()
    with torch.no_grad():
        layer.weight.fill_(.5)
    x = torch.zeros(1, 3, 8, 8)
    def draw(seed):
        return layer(x, noise_mode='stochastic', generator=torch.Generator().manual_seed(seed))
    assert torch.equal(draw(4), draw(4))
    assert not torch.equal(draw(4), draw(5))
    assert torch.equal(layer(x, noise_mode='deterministic'), x)
    assert not layer.training


def test_shared_paper_networks_and_direction_conditioning():
    model = DefectGAN(base_channels_g=4, base_channels_d=4, num_res_blocks=1,
                      architecture='defectgan_paper_baseline', num_disc_layers=3)
    assert model.G_n2d is model.G_d2n
    assert model.D_defect is model.D_normal
    assert len(list(model.generator_parameters())) == len({id(p) for p in model.generator_parameters()})
    model.eval()
    x = torch.zeros(1, 3, 32, 32)
    seen = []
    hook = model.G_n2d.register_forward_pre_hook(lambda module,args: seen.append(args[1]))
    model.defacement(x)
    model.restoration(x, torch.ones(1, 1))
    hook.remove()
    assert seen[0].eq(1).all() and seen[1].eq(0).all()
    assert model.G_n2d.encoder_stride == 4


def test_spatial_conditioning_and_scalar_noise():
    model = DefectGAN(base_channels_g=4, base_channels_d=4, num_res_blocks=1,
                      architecture='defectgan_paper_baseline', num_disc_layers=3).eval()
    x = torch.rand(1, 3, 32, 32)*2-1
    result = model.defacement(x, spatial_map=torch.zeros(1, 1, 32, 32))
    assert result['output'].shape == x.shape
    assert torch.isfinite(result['output']).all()
    assert all(m.weight.numel()==1 for m in model.G_n2d.modules() if isinstance(m, AdaptiveNoiseInjection))
    with pytest.raises(ValueError):
        model.defacement(x, spatial_map=torch.zeros(1, 2, 32, 32))


def test_wasserstein_sign_and_patch_scaling():
    from src.losses.gan_losses import WGANLoss
    loss = WGANLoss()
    real, fake = [torch.full((2,1,3,3), 3.)], [torch.full((2,1,3,3), 1.)]
    assert loss.discriminator_loss(real, fake).item() == -2.
    assert loss.generator_loss(fake).item() == -1.


def test_gradient_penalty_is_per_sample_and_reaches_critic_only():
    from src.losses.gan_losses import PaperDefectGANLoss
    criterion = PaperDefectGANLoss()
    class LinearCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
        def forward(self, x):
            return [(self.weight*x.sum((1,2,3), keepdim=True), x.new_zeros(x.shape[0],1))]
    critic = LinearCritic()
    real = torch.zeros(2, 3, 2, 2)
    fake = torch.ones_like(real, requires_grad=True)
    gp = criterion.gradient_penalty(critic, real, fake)
    assert gp.item() == pytest.approx((2*(12**.5)-1)**2)
    gp.backward()
    assert critic.weight.grad.abs() > 0
    assert fake.grad is None


def test_paper_all_loss_terms_and_weights():
    from src.losses.gan_losses import PaperDefectGANLoss
    criterion = PaperDefectGANLoss()
    image = torch.zeros(1,3,4,4)
    def result(value, mask):
        return {'output': torch.full_like(image, value, requires_grad=True),
                'mask': torch.full((1,1,4,4), mask, requires_grad=True)}
    fd, fn = result(.2,.2), result(.3,.3)
    rn, rd = result(.4,.4), result(.5,.5)
    predictions = [(torch.ones(1,1,2,2,requires_grad=True), torch.zeros(1,1,requires_grad=True))]
    total, logs = criterion.compute_generator_loss(image,image,fd,fn,rn,rd,None,None,
                         predictions,predictions,torch.ones(1,1),torch.zeros(1,1))
    # Both directions are averaged: adv=-1, cls=ln2, rec=.45,
    # spatial cycle=.2, region=(.2+.4+.3+.5)/2=.7.
    assert total.item() == pytest.approx(-1+5*torch.log(torch.tensor(2.)).item()+5*.45+5*.2+.7)
    total.backward()
    assert rn['output'].grad is not None
    assert fd['mask'].grad.abs().sum() > 0
    assert rd['mask'].grad.abs().sum() > 0
    assert 'g_sd_cyc' in logs and 'g_sd_con' in logs
