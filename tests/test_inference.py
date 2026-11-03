"""Inference contracts: coverage, CPU memory, controls and lossless RGB export."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import run_sliding_window_inference, process_single_image, load_model, parse_args


class IdentityModel(torch.nn.Module):
    num_classes = 1

    def defacement(self, x, c=None, **kwargs):
        assert not torch.is_grad_enabled()
        return {"output": x, "mask": torch.ones_like(x[:, :1])}

    restoration = defacement


@pytest.mark.parametrize("shape", [(16, 16), (31, 37), (7, 9), (7, 27), (23, 8)])
def test_coverage_and_original_dimensions(shape):
    """Every pixel survives padding/blending, including one small dimension."""
    x = torch.rand(1, 3, *shape) * 2 - 1
    out, alpha = run_sliding_window_inference(IdentityModel().eval(), x,
        "defacement", 16, 8, torch.device("cpu"))
    assert out.shape == x.shape
    assert alpha.shape == (1, 1, *shape)
    assert out.device.type == "cpu"
    assert torch.isfinite(out).all() and torch.isfinite(alpha).all()
    assert torch.allclose(out, x, atol=2e-6)
    assert torch.allclose(alpha, torch.ones_like(alpha))


@pytest.mark.parametrize("patch,stride", [(16, 17), (16, 0), (0, 1), (16, -1)])
def test_invalid_stride_rejected(patch, stride):
    with pytest.raises(ValueError):
        run_sliding_window_inference(IdentityModel(), torch.zeros(1,3,19,19),
            "defacement", patch, stride, torch.device("cpu"))


class WhiteModel(IdentityModel):
    def defacement(self, x, c=None, **kwargs):
        return {"output": torch.ones_like(x), "mask": torch.ones_like(x[:, :1])}


def test_allowed_roi_and_protected_region_preserved_exactly():
    x = torch.rand(1,3,21,25) * 2 - 1
    roi = torch.zeros(1,1,21,25)
    roi[..., 3:18, 2:20] = 1
    protect = torch.zeros_like(roi)
    protect[..., 8:11, 6:13] = 1
    out, alpha = run_sliding_window_inference(WhiteModel().eval(), x,
        "defacement", 16, 8, torch.device("cpu"), allowed_roi=roi, protected_roi=protect)
    allowed = (roi.bool() & ~protect.bool()).expand_as(x)
    assert torch.equal(out[~allowed], x[~allowed])
    assert torch.equal(out[allowed], torch.ones_like(out[allowed]))
    assert torch.equal(alpha[~allowed[:, :1]], torch.zeros_like(alpha[~allowed[:, :1]]))


def test_export_preserves_forbidden_rgb_bytes(tmp_path):
    arr = np.random.default_rng(3).integers(0,256,(9,23,3),dtype=np.uint8)
    source = tmp_path / "source.png"
    target = tmp_path / "output.png"
    Image.fromarray(arr).save(source)
    roi = torch.zeros(1,1,9,23)
    process_single_image(WhiteModel().eval(), source, target, "defacement",16,8,
        True,torch.device("cpu"),allowed_roi=roi)
    with Image.open(target) as img:
        assert img.mode == "RGB" and np.array_equal(np.asarray(img), arr)
    with Image.open(tmp_path / "output_alpha.png") as alpha:
        assert alpha.mode == "L"


def test_spatial_map_sliced_without_becoming_export_roi():
    class RecordingModel(WhiteModel):
        def defacement(self, x, c=None, **kwargs):
            assert kwargs["spatial_map"].shape[-2:] == x.shape[-2:]
            return super().defacement(x,c,**kwargs)
    x = torch.zeros(1,3,9,23)
    control = torch.zeros(1,1,9,23)
    out,_ = run_sliding_window_inference(RecordingModel().eval(),x,"defacement",
        16,8,torch.device("cpu"),spatial_map=control)
    assert torch.equal(out,torch.ones_like(out))


def test_noise_seed_contract_with_nonzero_noise_weights():
    from src.models.defect_gan import DefectGAN
    from src.models.generator import AdaptiveNoiseInjection
    torch.set_num_threads(1)
    model = DefectGAN(base_channels_g=4,base_channels_d=4,num_res_blocks=1,num_disc_scales=1).eval()
    for module in model.modules():
        if isinstance(module, AdaptiveNoiseInjection):
            with torch.no_grad():
                module.weight.fill_(0.3)
    x = torch.zeros(1,3,16,16)
    def infer(seed,mode):
        return run_sliding_window_inference(model,x,"defacement",16,8,
            torch.device("cpu"),seed=seed,noise_mode=mode)[0]
    assert torch.equal(infer(7,"stochastic"),infer(7,"stochastic"))
    assert not torch.equal(infer(7,"stochastic"),infer(8,"stochastic"))
    assert torch.equal(infer(7,"deterministic"),infer(8,"deterministic"))
    assert not model.training


@pytest.mark.parametrize("architecture", ["defectgan_custom_legacy", "defectgan_paper_baseline", None])
def test_checkpoint_architecture_roundtrip(tmp_path, architecture):
    from src.models.defect_gan import DefectGAN
    config = {"model": {"architecture": architecture,
        "generator": {"base_channels": 4,"num_resblocks": 1,"num_classes": 1},
        "discriminator": {"base_channels": 4,"num_scales": 1,"num_layers": 3}}}
    if architecture is None:
        del config["model"]["architecture"]
    model = DefectGAN(base_channels_g=4,base_channels_d=4,num_res_blocks=1,
        num_disc_scales=1,num_disc_layers=3,architecture=architecture or "defectgan_custom_legacy").eval()
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state":model.state_dict(),"config":config,"epoch":0},path)
    loaded = load_model(path,torch.device("cpu"))
    for key,value in model.state_dict().items():
        assert torch.equal(value,loaded.state_dict()[key])
    x = torch.zeros(1,3,32,32)
    a,_ = run_sliding_window_inference(model,x,"defacement",32,16,torch.device("cpu"))
    b,_ = run_sliding_window_inference(loaded,x,"defacement",32,16,torch.device("cpu"))
    assert torch.equal(a,b) and not loaded.training


@pytest.mark.parametrize("control", [torch.ones(1,1,4,5),torch.full((1,1,9,9),float("nan")),
                                    torch.full((1,1,9,9),2.)])
def test_invalid_controls_rejected(control):
    with pytest.raises(ValueError):
        run_sliding_window_inference(IdentityModel(),torch.zeros(1,3,9,9),"defacement",
            16,8,torch.device("cpu"),allowed_roi=control)


def test_cli_exposes_seed_controls_and_rejects_conflicting_sources():
    args = parse_args(["--checkpoint","c.pth","--input","in.png","--output","o.png",
        "--seed","9","--noise","stochastic","--roi","r.png","--protect","p.png",
        "--spatial-map","s.png","--save-alpha"])
    assert args.seed == 9 and args.noise == "stochastic" and args.save_mask
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint","c.pth","--input","in.png","--input-dir","images"])


def test_legacy_restoration_preserves_historical_category_conditioning():
    class LegacyModel(IdentityModel):
        architecture = "defectgan_custom_legacy"
        def restoration(self,x,c=None,**kwargs):
            assert torch.equal(c,torch.ones_like(c))
            return super().restoration(x,c,**kwargs)
    run_sliding_window_inference(LegacyModel(),torch.zeros(1,3,16,16),
        "restoration",16,8,torch.device("cpu"))


@pytest.mark.parametrize("alpha_collision", [False, True])
def test_export_cannot_overwrite_input(tmp_path,alpha_collision):
    source = tmp_path / ("out_alpha.png" if alpha_collision else "out.png")
    Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(source)
    before = source.read_bytes()
    with pytest.raises(ValueError,match="overwrite"):
        process_single_image(IdentityModel(),source,tmp_path/"out.png","defacement",
            16,8,True,torch.device("cpu"))
    assert source.read_bytes() == before


def test_spatial_map_respects_restoration_category():
    """A Scratch control map must not switch restoration back to defacement."""
    from src.models.defect_gan import DefectGAN
    torch.set_num_threads(1)
    model = DefectGAN(base_channels_g=4, base_channels_d=4, num_res_blocks=1,
                      architecture="defectgan_paper_baseline", num_disc_layers=3).eval()
    seen = []
    handle = model.G_n2d.dec2.spade.register_forward_pre_hook(
        lambda module, args: seen.append(args[1].detach().clone()))
    x = torch.zeros(1, 3, 32, 32)
    spatial_map = torch.full((1, 1, 32, 32), .75)
    model.defacement(x, spatial_map=spatial_map)
    model.restoration(x, spatial_map=spatial_map)
    handle.remove()
    assert torch.equal(seen[0], spatial_map)
    assert torch.count_nonzero(seen[1]) == 0
