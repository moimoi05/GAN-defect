"""Dataset contracts: portable paths, rejection, deterministic evaluation, read-only audit."""
import csv
import hashlib
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.datasets.f971_dataset import DefectGANDataset
from scripts import validate_dataset as validator

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_DIR", ROOT))


def write_manifest(tmp_path, path='patches/clean.png', split='validation'):
    for domain in ('clean', 'defect'):
        p = tmp_path / 'patches' / f'{domain}.png'
        p.parent.mkdir(exist_ok=True)
        Image.new('RGB', (16, 16), (20, 50, 100)).save(p)
    manifest = tmp_path / 'val.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename','domain','split','source_image','patch_path'])
        writer.writeheader()
        for domain in ('clean', 'defect'):
            writer.writerow(dict(filename=f'{domain}.png', domain=domain, split=split,
                                 source_image=f'{domain}.jpg', patch_path=path if domain=='clean' else 'patches/defect.png'))
    return manifest


def test_windows_manifest_separators_resolve_with_explicit_data_root(tmp_path):
    manifest = write_manifest(tmp_path, 'patches\\clean.png')
    ds = DefectGANDataset(manifest, data_root=tmp_path, split='validation', image_size=16)
    assert ds[0]['normal'].shape == (3,16,16)


@pytest.mark.parametrize('bad_path', ['../outside.png', '/etc/passwd', 'D:/dataset/clean.png'])
def test_manifest_rejects_paths_outside_dataset(tmp_path, bad_path):
    manifest = write_manifest(tmp_path, bad_path)
    with pytest.raises(ValueError, match='relative|escape|absolute'):
        DefectGANDataset(manifest, data_root=tmp_path, split='validation', image_size=16)


def test_manifest_cannot_silently_load_train_rows_as_validation(tmp_path):
    manifest = write_manifest(tmp_path, split='train')
    with pytest.raises(ValueError, match='split'):
        DefectGANDataset(manifest, data_root=tmp_path, split='validation', image_size=16)


def test_non_rgb_input_is_rejected_without_conversion(tmp_path):
    manifest = write_manifest(tmp_path)
    Image.new('L', (16,16)).save(tmp_path/'patches/clean.png')
    ds = DefectGANDataset(manifest, data_root=tmp_path, split='validation', image_size=16)
    with pytest.raises(ValueError, match='RGB'):
        ds[0]


def test_validation_is_deterministic_and_normalized(tmp_path):
    manifest = write_manifest(tmp_path)
    ds = DefectGANDataset(manifest, data_root=tmp_path, split='validation', image_size=16)
    a,b = ds[0],ds[0]
    assert torch.equal(a['normal'], b['normal'])
    assert a['normal'].min() >= -1 and a['normal'].max() <= 1


def test_baseline_validation_read_only():
    manifests = list((ROOT/'training/manifests').glob('*.csv'))
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}
    result = validator.validate_baseline(DATA_ROOT, ROOT/"training/manifests")
    assert result['total'] == 114
    assert result['clean_replacements'] == 39
    assert result['source_count'] == 92
    assert before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}


def test_manifest_generation_requires_separate_output_directory():
    with pytest.raises(ValueError, match='separate|protected'):
        validator.generate_manifests(DATA_ROOT, DATA_ROOT/'training/manifests')

@pytest.mark.parametrize('args', [['--generate-manifests'], ['--output-dir', 'unused']])
def test_validator_cli_rejects_incomplete_generation_request(args):
    with pytest.raises(SystemExit) as exc:
        validator.main(args)
    assert exc.value.code == 2


def test_generator_refuses_to_overwrite_existing_output(tmp_path):
    (tmp_path/'existing.csv').write_text('preserve me')
    with pytest.raises(ValueError, match='empty'):
        validator.generate_manifests(DATA_ROOT, tmp_path)
    assert (tmp_path/'existing.csv').read_text() == 'preserve me'


@pytest.mark.parametrize('mode,size,fmt', [('L',(1024,1024),'PNG'),('RGB',(20,30),'PNG'),('RGB',(1024,1024),'JPEG')])
def test_validator_rejects_wrong_image_contract(tmp_path,mode,size,fmt):
    p=tmp_path/'wrong.png'
    Image.new(mode,size).save(p,format=fmt)
    with pytest.raises(ValueError):
        validator.validate_image_properties([{'full_path':p}])


def test_validator_rejects_source_leakage():
    with pytest.raises(RuntimeError,match='leakage'):
        validator.check_data_leakage([{'source_image':'same.jpg','split':'train'},
                                      {'source_image':'same.jpg','split':'test'}])

def test_explicit_data_root_never_falls_back_to_manifest_parent(tmp_path):
    manifest_dir = tmp_path/'training/manifests'
    manifest_dir.mkdir(parents=True)
    original = write_manifest(tmp_path)
    manifest = manifest_dir/'val.csv'
    manifest.write_bytes(original.read_bytes())
    empty_root = tmp_path/'empty_mount'
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match='data_root'):
        DefectGANDataset(manifest, data_root=empty_root, split='validation', image_size=16)


def test_images_resolve_from_data_mount_when_manifest_is_in_code_tree(tmp_path):
    mount = tmp_path/'dataset'
    mount.mkdir()
    original = write_manifest(mount)
    code = tmp_path/'code'
    code.mkdir()
    manifest=code/'val.csv'
    manifest.write_bytes(original.read_bytes())
    before=hashlib.sha256((mount/'patches/clean.png').read_bytes()).hexdigest()
    ds=DefectGANDataset(manifest,data_root=mount,split='validation',image_size=16)
    assert ds[0]['normal'].shape==(3,16,16)
    assert before==hashlib.sha256((mount/'patches/clean.png').read_bytes()).hexdigest()

def test_explicit_manifest_generation_writes_only_separate_output(tmp_path):
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (DATA_ROOT/'training/manifests').glob('*.csv')}
    validator.generate_manifests(DATA_ROOT,tmp_path/'generated')
    assert len(list((tmp_path/'generated').glob('*.csv')))==3
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (DATA_ROOT/'training/manifests').glob('*.csv')}


def test_manifest_consistency_mismatch_fails_closed(tmp_path):
    """Manifest consistency check: differing copies between image and DATA_DIR fail-closed."""
    from src.datasets.f971_dataset import verify_manifest_consistency
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rel = Path("training/manifests/val.csv")
    (data_dir / rel).parent.mkdir(parents=True)
    (tmp_path / rel).parent.mkdir(parents=True)
    # Write inconsistent contents
    (data_dir / rel).write_text("patch_path,domain,split\npatches/clean.png,clean,validation\n")
    (tmp_path / rel).write_text("patch_path,domain,split\npatches/other.png,clean,validation\n")

    with pytest.raises(ValueError, match="Manifest consistency check FAILED"):
        verify_manifest_consistency(rel, data_root=data_dir)


def test_manifest_consistency_matching_passes(tmp_path):
    """Manifest consistency check: identical copies across candidates resolve successfully."""
    from src.datasets.f971_dataset import verify_manifest_consistency
    # Test using actual repository manifests across project and data_root
    resolved = verify_manifest_consistency("training/manifests/val.csv", data_root=ROOT)
    assert resolved.exists()
    assert resolved.name == "val.csv"

    # Test custom separate directory with consistent copies
    sub1 = tmp_path / "img_dir"
    sub2 = tmp_path / "data_dir"
    sub1.mkdir()
    sub2.mkdir()
    content = "patch_path,domain,split\npatches/clean.png,clean,validation\n"
    (sub1 / "test_val.csv").write_text(content)
    (sub2 / "test_val.csv").write_text(content)
    resolved2 = verify_manifest_consistency(sub1 / "test_val.csv", data_root=sub2)
    assert resolved2.exists()
    assert resolved2.read_text() == content
