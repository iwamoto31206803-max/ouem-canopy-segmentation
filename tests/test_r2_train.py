"""Small R1a PASS fixtures and CPU models only. No imagery or weight downloads."""
import copy
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import r2_train as trainer
from test_r1_dataset import dataset, seal


@pytest.fixture
def prepared(dataset):
    root, manifest = dataset
    data_root = root/'data'
    data_root.mkdir()
    for split in ('train', 'val'):
        shutil.move(str(root/split), data_root/split)
    root = data_root
    sample = copy.deepcopy(manifest['samples'][-1])
    sample.update(image_id=3, oam_id='synthetic-oam-3', products={})
    h, w = 5, 7
    transform = from_origin(300000, 10000, .5, .5)
    bounds = [300000, 10000-h*.5, 300000+w*.5, 10000]
    sample['grid'] = dict(crs='EPSG:3395', transform=list(transform)[:6], bounds=bounds, shape=[h, w])
    source = sample['source']
    source.update(original_source_id='synthetic-scene-3', image_sha256='f'*64,
                  footprint_wgs84=list(trainer.r1.transform_bounds('EPSG:3395', 4326, *bounds)))
    for key in ('category_review', 'validity_review', 'independence_review', 'external_independence_review', 'r1_review'):
        sample[key]['source_hashes'] = {k: source[k] for k in ('image_sha256', 'annotation_sha256')}
    values = np.arange(h*w, dtype=np.uint8).reshape(h, w)
    mask = (values % 2)[None]
    arrays = {'rgb.tif': np.repeat(values[None], 3, 0), 'canopy_mask.tif': mask,
              'canopy_fraction.tif': mask.astype(np.float32), 'valid_fraction.tif': np.ones((1,h,w),np.float32)}
    for name, array in arrays.items():
        path = root/'val'/'3'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=array.shape[0],
                           dtype=array.dtype, crs='EPSG:3395', transform=transform) as dst:
            dst.write(array)
        sample['products'][name] = dict(path=path.relative_to(root).as_posix(), sha256=trainer.r1.sha(path))
    manifest['samples'].append(sample)
    seal(manifest)
    path = root/'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    assert trainer.r1.verify(root, manifest)['acceptance'] == 'PASS'
    return root, manifest, path


def open_data(prepared):
    root, manifest, path = prepared
    return trainer.R1Data(root, path, manifest['dataset_id'], trainer.r1.sha(path))


class Tiny(torch.nn.Module):
    """Satlas-style return contract with ordinary CE and stochastic CPU layers."""
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 2, 1)
        self.drop = torch.nn.Dropout(.2)

    def forward(self, images, target):
        logits = self.head(self.drop(images)) * (1 + .01*random.random() + .01*np.random.random())
        return logits.softmax(1), torch.nn.functional.cross_entropy(logits, target)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_schedule_and_aligned_uniform_patches(prepared):
    data = open_data(prepared)
    plan = trainer.make_plan(data, patch_hw=(2,3), steps=7)
    assert plan == trainer.make_plan(data, patch_hw=(2,3), steps=7)
    other = trainer.make_plan(data, seed=1, patch_hw=(2,3), steps=7)
    assert other['train'] != plan['train'] and other['val'] == plan['val']
    assert len(trainer.make_plan(data, patch_hw=(2,2), steps=23)['train']) == 23
    assert {s['image_id'] for s in plan['train']} == {0,1}
    assert {s['image_id'] for s in plan['val']} == {2,3}
    for start in range(0, 7, 2):
        ids = [s['image_id'] for s in plan['train'][start:start+2]]
        assert len(ids) == len(set(ids))
    for rid, value in [(0,0), (1,1)]:
        item = next(s for s in plan['train'] if s['image_id'] == rid)
        x, y, _ = data.patch(item, 'train')
        assert x.shape == (1,3,2,3) and y.shape == (1,2,3)
        assert torch.all(y == value)
    item = dict(image_id=3, oam_id=data.samples[3]['oam_id'], crop_xywh=[2,1,3,2])
    x, y, crops = data.patch(item, 'val')
    full = trainer.r1.load_sample(data.verified, 3, 'val')
    for name, array in crops.items():
        assert np.array_equal(array, full[name][:,1:3,2:5])
    assert torch.equal(x[0], torch.tensor(crops['rgb.tif'], dtype=torch.float32)/255)
    assert torch.equal(y, torch.tensor(crops['canopy_mask.tif'], dtype=torch.long))
    with pytest.raises(ValueError): data.patch(item, 'train')
    item['crop_xywh'][0] = 99
    with pytest.raises(ValueError): data.patch(item, 'val')


@pytest.mark.parametrize('ids', [[0], [0,1,2], [0,1,2,3]])
def test_arbitrary_dataset_size(prepared, ids):
    root, m, path = prepared
    m['samples'] = [s for s in m['samples'] if s['image_id'] in ids]
    seal(m)
    path.write_text(json.dumps(m), encoding='utf-8')
    data = open_data(prepared)
    assert len(data.samples) == len(ids)
    assert len(trainer.make_plan(data, patch_hw=(2,2), steps=1)['train']) == 1


@pytest.mark.parametrize('change', [
    lambda m: m['samples'][0]['r1_review'].update(status='HOLD'),
    lambda m: m['samples'][0]['rights'].update(original_source_license='CC-BY-NC-4.0'),
    lambda m: m['samples'][0]['source'].pop('original_source_id'),
    lambda m: m['samples'][0].update(official_split='test'),
    lambda m: m['samples'][0].update(validation_fold=0),
    lambda m: m['samples'][0].update(source_generation='HF'),
])
def test_r1a_rejections_preserved(prepared, change):
    root, m, path = prepared
    change(m)
    seal(m)
    path.write_text(json.dumps(m), encoding='utf-8')
    with pytest.raises(ValueError): open_data(prepared)


def test_identity_hash_crop_and_settings_rejection(prepared):
    root, m, path = prepared
    with pytest.raises(ValueError): trainer.R1Data(root, path, 'other', trainer.r1.sha(path))
    with pytest.raises(ValueError): trainer.R1Data(root, path, m['dataset_id'], '0'*64)
    data = open_data(prepared)
    for args in [dict(patch_hw=(100,100)), dict(steps=0), dict(seed=True), dict(val_ids=[0]),
                 dict(val_ids=[999]), dict(val_ids=[2,2]), dict(val_origin=(99,0)), dict(lr=float('nan'))]:
        with pytest.raises(ValueError): trainer.make_plan(data, patch_hw=args.pop('patch_hw', (2,2)), **args)
    rgb = root/data.samples[0]['products']['rgb.tif']['path']
    rgb.write_bytes(b'changed synthetic product')
    with pytest.raises(ValueError): data.patch(trainer.make_plan(data, patch_hw=(2,2))['train'][0], 'train')


def test_one_step_and_exact_cpu_continuation(prepared, tmp_path):
    data = open_data(prepared)
    plan = trainer.make_plan(data, patch_hw=(2,2), steps=4)
    whole = trainer.train(data, plan, Tiny, tmp_path/'whole')
    first = trainer.train(data, plan, Tiny, tmp_path/'first', stop_after=1)
    assert first['completed_steps'] == 1 and len(first['history']) == 1
    checkpoint = tmp_path/'first/checkpoint.pth'
    initial_state = trainer.safe_load(checkpoint, first['checkpoint_sha256'])
    torch.manual_seed(plan['config']['seed'])
    initial = Tiny().state_dict()
    assert any(not torch.equal(initial[k], initial_state['model'][k]) for k in initial)
    resumed = trainer.train(data, plan, Tiny, tmp_path/'resumed', resume=checkpoint,
                            resume_sha256=first['checkpoint_sha256'])
    a = trainer.safe_load(tmp_path/'whole/checkpoint.pth', whole['checkpoint_sha256'])
    b = trainer.safe_load(tmp_path/'resumed/checkpoint.pth', resumed['checkpoint_sha256'])
    assert a['plan'] == b['plan'] and a['history'] == b['history']
    for key in a['model']: assert torch.equal(a['model'][key], b['model'][key])
    for i in a['optimizer']['state']:
        for k, value in a['optimizer']['state'][i].items():
            assert torch.equal(value, b['optimizer']['state'][i][k])
    assert torch.equal(a['cpu_rng'], b['cpu_rng'])
    assert a['python_rng'] == b['python_rng'] and a['numpy_rng'] == b['numpy_rng']
    images, target, _ = data.patch(plan['val'][0], 'val')
    predictions = []
    for state in (a, b):
        model = Tiny().eval()
        model.load_state_dict(state['model'])
        trainer._restore_rng(state)
        with torch.inference_mode():
            predictions.append(model(images, target)[0])
    assert torch.equal(*predictions)
    assert trainer.r1.sha(checkpoint) == first['checkpoint_sha256']
    with pytest.raises(FileExistsError): trainer.train(data, plan, Tiny, tmp_path/'first')
    with pytest.raises(ValueError): trainer.train(data, plan, Tiny, tmp_path/'outside', stop_after=5)
    with pytest.raises(ValueError): trainer.train(data, plan, Tiny, tmp_path/'bad-hash', resume=checkpoint, resume_sha256='0'*64)
    changed = trainer.make_plan(data, patch_hw=(2,2), steps=4, seed=1)
    with pytest.raises(ValueError): trainer.train(data, changed, Tiny, tmp_path/'changed', resume=checkpoint, resume_sha256=first['checkpoint_sha256'])
    altered = copy.deepcopy(plan)
    altered['train'][0]['crop_xywh'][0] = 999
    with pytest.raises(ValueError): trainer.train(data, altered, Tiny, tmp_path/'altered')
    with pytest.raises(ValueError): trainer.train(data, plan, Tiny, data.verified.root/'output')
    # A newly valid R1a manifest still cannot replace the checkpoint's dataset.
    root, manifest, path = prepared
    manifest['samples'][0]['rights']['attribution'] = 'different synthetic attribution'
    seal(manifest)
    path.write_text(json.dumps(manifest), encoding='utf-8')
    other_data = open_data(prepared)
    other_plan = trainer.make_plan(other_data, patch_hw=(2,2), steps=4)
    with pytest.raises(ValueError):
        trainer.train(other_data, other_plan, Tiny, tmp_path/'changed-data', resume=checkpoint,
                      resume_sha256=first['checkpoint_sha256'])


@pytest.mark.parametrize('case', ['kind', 'history', 'optimizer', 'completed'])
def test_inconsistent_checkpoint_rejected(prepared, tmp_path, case):
    data = open_data(prepared)
    plan = trainer.make_plan(data, patch_hw=(2,2), steps=2)
    record = trainer.train(data, plan, Tiny, tmp_path/'first', stop_after=1)
    state = trainer.safe_load(tmp_path/'first/checkpoint.pth', record['checkpoint_sha256'])
    if case == 'kind': state['kind'] = 'OLD_G2'
    if case == 'history': state['history'][0]['crop_xywh'][0] = 999
    if case == 'optimizer': state['optimizer']['param_groups'][0]['lr'] = .1
    if case == 'completed': state['completed_steps'] = 2
    path = tmp_path/'altered.pth'
    sha = trainer.save_state(path, state)
    with pytest.raises(ValueError):
        trainer.train(data, plan, Tiny, tmp_path/'rejected', resume=path, resume_sha256=sha)
    assert not (tmp_path/'rejected').exists()


def test_nonfinite_step_reports_failure(prepared, tmp_path):
    class Bad(Tiny):
        def forward(self, images, target):
            probability, loss = super().forward(images, target)
            return probability, loss * float('nan')
    data = open_data(prepared)
    plan = trainer.make_plan(data, patch_hw=(2,2), steps=1)
    with pytest.raises(ValueError, match='output/loss'):
        trainer.train(data, plan, Bad, tmp_path/'failed')
    result = json.loads((tmp_path/'failed/result.json').read_text())
    assert result['status'] == 'FAIL' and result['completed_steps'] == 0
    assert not (tmp_path/'failed/checkpoint.pth').exists()


def test_failure_preserves_source_checkpoint_and_partial(prepared, tmp_path, monkeypatch):
    data = open_data(prepared)
    plan = trainer.make_plan(data, patch_hw=(2,2), steps=2)
    first = trainer.train(data, plan, Tiny, tmp_path/'first', stop_after=1)
    source = tmp_path/'first/checkpoint.pth'
    def broken(path, payload):
        path.write_bytes(b'partial synthetic checkpoint')
        raise OSError('injected write failure')
    monkeypatch.setattr(trainer, 'save_state', broken)
    with pytest.raises(OSError):
        trainer.train(data, plan, Tiny, tmp_path/'failed', resume=source, resume_sha256=first['checkpoint_sha256'])
    assert (tmp_path/'failed/checkpoint.pth.partial').exists()
    assert not (tmp_path/'failed/checkpoint.pth').exists()
    assert json.loads((tmp_path/'failed/result.json').read_text())['status'] == 'FAIL'
    assert trainer.r1.sha(source) == first['checkpoint_sha256']
