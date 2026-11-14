"""Synthetic 224px C1-style products, explicit research review, tiny CPU CE."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin, array_bounds
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import research_data as research
import r2_train as trainer
from test_r2_train import Tiny


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path/'patches'
    root.mkdir()
    samples = []
    for i, (rid, split) in enumerate(research.TARGETS.items()):
        full = from_origin(i*1000, 2000, .5, .5)
        transform = full*rasterio.Affine.translation(10, 20)
        fraction = np.full((1, 224, 224), i % 2, np.float32)
        arrays = {'rgb.tif': np.full((3, 224, 224), i*30, np.uint8),
                  'canopy_fraction.tif': fraction, 'canopy_mask.tif': fraction.astype(np.uint8)}
        products = {}
        for name, array in arrays.items():
            path = root/split/str(rid)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, 'w', driver='GTiff', count=array.shape[0],
                               height=224, width=224, dtype=array.dtype, crs='EPSG:3395',
                               transform=transform) as dst:
                dst.write(array)
            products[name] = dict(path=path.relative_to(root).as_posix(), sha256=research.r1.sha(path))
        oam = f'SYNTHETIC-{rid}'
        samples.append(dict(image_id=rid, oam_id=oam, split=split,
            validation_fold=0 if split == 'val' else 1,
            source_hashes={'rgb': 'a'*64, 'annotation': 'b'*64},
            source_metadata='https://example.invalid/SYNTHETIC_meta.json',
            rights=dict(row_license='CC-BY 4.0', image_license='CC-BY-4.0',
                annotation_license='CC BY 4.0', producer='SYNTHETIC', attribution='SYNTHETIC test only',
                stac_sha256='c'*64, annotation_terms='https://example.invalid/SYNTHETIC',
                stac_url='https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+oam),
            full_grid=dict(shape=[400, 400], affine=list(full)[:6]), crop_xywh=[10,20,224,224],
            patch_grid=dict(shape=[224,224], affine=list(transform)[:6], crs='EPSG:3395',
                            bounds=list(array_bounds(224,224,transform))),
            selected_black_support_cells=0, products=products))
    plan = dict(schema='ouem-exploratory-crops-v1', purpose='PHASE_C1_RESEARCH_ONLY',
                source_generation=research.GENERATION, status='EXPLORATORY_PILOT_PRODUCTS_COMPLETE',
                policy=research.POLICY, r1_acceptance='UNCHANGED_HOLD', samples=samples)
    review = dict(schema='ouem-research-patch-review-v1', formal_r1='UNCHANGED_HOLD',
                  reviewer='SYNTHETIC ONLY', samples=[dict(image_id=rid,
                  decision='REVIEW_REQUIRED' if rid == 1054 else 'CLEAR_FOR_EXPLORATION',
                  reason='Synthetic review fixture') for rid in research.TARGETS])
    return root, plan, review


def open_data(pilot):
    root, plan, review = pilot
    path = root/'crop_plan.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    review['crop_plan_sha256'] = research.r1.sha(path)
    review_path = root.parent/'review.json'
    review_path.write_text(json.dumps(review), encoding='utf-8')
    return research.ResearchData(root, research.r1.sha(path), review_path, research.r1.sha(review_path))


def test_tiny_training_and_fixed_val(pilot, tmp_path):
    data = open_data(pilot)
    assert set(data.samples) == {2493, 3206, 1241, 205}
    plan = trainer.make_plan(data, steps=1)
    assert plan == trainer.make_plan(data, steps=1)
    assert plan['inputs']['purpose'] == 'EXPLORATORY_RESEARCH_ONLY'
    assert {s['image_id'] for s in plan['val']} == {1241, 205}
    before = {str(p): research.r1.sha(p) for p in data.root.rglob('*.tif')}
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = trainer.train(data, plan, Tiny, tmp_path/'training')
    finally:
        torch.set_num_threads(old_threads)
    assert result['status'] == 'CPU_TRAINING_CORE_PASS' and result['completed_steps'] == 1
    state = trainer.safe_load(tmp_path/'training/checkpoint.pth', result['checkpoint_sha256'])
    assert state['plan']['inputs'] == data.identity
    for item in plan['val']:
        image, target, arrays = data.patch(item, 'val')
        assert image.shape == (1,3,224,224) and target.shape == (1,224,224)
        assert image.dtype == torch.float32 and target.dtype == torch.long
        assert set(arrays) == set(research.PRODUCTS)
        assert torch.all(target == (1 if item['image_id'] == 1241 else 0))
    assert {str(p): research.r1.sha(p) for p in data.root.rglob('*.tif')} == before
    with pytest.raises(ValueError):
        data.patch(dict(image_id=1054, oam_id='SYNTHETIC-1054', crop_xywh=[0,0,224,224]), 'train')
    with pytest.raises(ValueError):
        data.patch(plan['val'][0], 'train')


@pytest.mark.parametrize('case', ['hash', 'split', 'fold', 'crop', 'label', 'grid', 'rights',
                                 'source', 'source_hash', 'review', 'missing_review', 'hf', 'path',
                                 'fraction', 'nan', 'dtype', 'mask', 'shape_type', 'attribution'])
def test_invalid_products_or_evidence(pilot, case):
    root, plan, review = pilot
    sample = plan['samples'][1]
    if case == 'hash': sample['products']['rgb.tif']['sha256'] = '0'*64
    elif case == 'split': sample['split'] = 'val'
    elif case == 'fold': sample['validation_fold'] = 1.0
    elif case == 'crop': sample['crop_xywh'][0] = 300
    elif case == 'rights': sample['rights']['image_license'] = 'unknown'
    elif case == 'source': sample['source_metadata'] = ''
    elif case == 'source_hash': sample['source_hashes']['annotation'] = 'bad'
    elif case == 'attribution': sample['rights']['attribution'] = ''
    elif case == 'shape_type': sample['patch_grid']['shape'][0] = 224.0
    elif case == 'review': review['samples'][1]['decision'] = 'PASS'
    elif case == 'missing_review': review['samples'].pop()
    elif case == 'hf': plan['source_generation'] = 'hf'
    elif case == 'path': sample['products']['rgb.tif']['path'] = '../rgb.tif'
    else:
        name = 'canopy_mask.tif' if case == 'label' else 'rgb.tif'
        if case in ('fraction', 'nan'): name = 'canopy_fraction.tif'
        product = sample['products'][name]
        path = root/product['path']
        if case == 'dtype':
            with rasterio.open(path) as src: profile = src.profile
            profile['dtype'] = 'uint16'
            with rasterio.open(path, 'w', **profile) as dst: dst.write(np.ones((3,224,224),np.uint16))
        else:
            with rasterio.open(path, 'r+') as dst:
                if case == 'label': dst.write(np.zeros((1,224,224),np.uint8))
                elif case in ('fraction', 'nan'):
                    dst.write(np.full((1,224,224),1.2 if case == 'fraction' else np.nan,np.float32))
                elif case == 'mask': dst.nodata = 30
                else: dst.transform = from_origin(1, 2, .5, .5)
        product['sha256'] = research.r1.sha(path)
    with pytest.raises(ValueError): open_data(pilot)


def test_pins_and_post_open_changes(pilot):
    data = open_data(pilot)
    item = trainer.make_plan(data, steps=1)['train'][0]
    with pytest.raises(ValueError):
        research.ResearchData(data.root, '0'*64, data.review_path, data.identity['research_review_sha256'])
    other = copy.deepcopy(item)
    other['crop_xywh'][0] = 1
    with pytest.raises(ValueError): data.patch(other, 'train')
    product = data.root/f"train/{item['image_id']}/canopy_mask.tif"
    with product.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError): data.patch(item, 'train')
    data.review_path.write_text('{}')
    with pytest.raises(ValueError): data.patch(item, 'train')
