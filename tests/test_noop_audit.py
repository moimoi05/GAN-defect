"""Regression contracts for coverage and differentiable composition."""
from pathlib import Path
import torch
import yaml
from src.datasets.full_patch_coverage import FullPatchCoverageDataset
from src.models.generator import compose_layers


def test_official_config_covers_every_manifest_patch():
    config = yaml.safe_load((Path(__file__).parents[1] / 'configs/defectgan_1024.yaml').read_text())
    assert config['training']['cover_all_manifest_rows_per_epoch'] is True
    class Base:
        normal_items = [dict(full_path=i, patch_path=i, source_image=i) for i in range(41)]
        defect_items = [dict(full_path=i, patch_path=i, source_image=i) for i in range(31)]
        def _load_image(self, path): return path
    dataset = FullPatchCoverageDataset(Base())
    pairs = [dataset[i] for i in range(len(dataset))]
    assert len(pairs) == 72
    assert {p['normal_path'] for p in pairs} == set(range(41))
    assert {p['defect_path'] for p in pairs} == set(range(31))


def test_composition_gradients_and_zero_mask_attenuation():
    x = torch.zeros(1, 3, 8, 8)
    foreground = torch.ones_like(x, requires_grad=True)
    alpha = torch.full((1, 1, 8, 8), .2, requires_grad=True)
    output = compose_layers(x, foreground, alpha)
    gf, ga = torch.autograd.grad(output.sum(), (foreground, alpha))
    assert torch.allclose(gf, alpha.expand_as(gf))
    assert torch.allclose(ga, torch.full_like(ga, 3))
    assert torch.equal(compose_layers(x, foreground, torch.zeros_like(alpha)), x)
