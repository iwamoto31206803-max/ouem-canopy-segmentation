"""Variable source pools and reviewed window schedules, synthetic pixels only."""
import copy
from collections import Counter
import json

import numpy as np
import pytest
import rasterio
import torch

from test_research_data import pilot, Tiny
from test_research_patches import inputs
import research_data as reader
import research_patches as patches
import r2_train as trainer


def pool(pilot, tmp_path, count):
    old_root, template, _ = pilot
    root = tmp_path/f'pool-{count}'
    root.mkdir()
    samples, candidates, reviews = [], [], []
    for i in range(count):
        sample = copy.deepcopy(template['samples'][i % 5])
        if i >= 5:
            sample['image_id'] = 99000+i
            sample['oam_id'] = f'SYNTHETIC-{99000+i}'
            sample['rights']['stac_url'] = 'https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+sample['oam_id']
        rid = sample['image_id']
        sample['source_grid'] = dict(crs='EPSG:3395', affine=sample['full_grid']['affine'])
        sample['placement_count'] = 100
        candidates.append({k: copy.deepcopy(sample[k]) for k in ('image_id','oam_id','split','validation_fold',
                           'source_hashes','source_metadata','placement_count')})
        candidates[-1]['output_shape'] = [400,400]
        for window in range(1 if sample['split'] == 'val' else 2):
            record = copy.deepcopy(sample)
            record['crop_xywh'] = [10+window,20,224,224]
            transform = rasterio.Affine(*record['full_grid']['affine'])*rasterio.Affine.translation(10+window,20)
            record['patch_grid']['affine'] = list(transform)[:6]
            record['patch_grid']['bounds'] = list(rasterio.transform.array_bounds(224,224,transform))
            record['products'] = {}
            for name, source in sample['products'].items():
                with rasterio.open(old_root/source['path']) as src:
                    array, profile = src.read(), src.profile
                path = root/f"{sample['split']}/{rid}/crop-{10+window}-20/{name}"
                path.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.open(path,'w',**dict(profile,transform=transform)) as dst:
                    dst.write(array)
                record['products'][name] = dict(path=path.relative_to(root).as_posix(),sha256=reader.r1.sha(path))
            samples.append(record)
            reviews.append(dict(image_id=rid,crop_xywh=record['crop_xywh'],
                decision='REVIEW_REQUIRED' if rid == 1054 and window == 0 else 'CLEAR_FOR_EXPLORATION',
                reason='SYNTHETIC window review'))
    candidate = dict(schema='ouem-research-candidates-v1',source_generation=patches.GENERATION,
                     policy=patches.POLICY,phase_b_sha256='a'*64,
                     metadata_sha256={'train_meta.json':'b'*64,'test_meta.json':'c'*64},
                     candidates=candidates)
    candidate_path = root/'candidates.json'
    candidate_path.write_text(json.dumps(candidate))
    candidate_sha = reader.r1.sha(candidate_path)
    plan = dict(template,schema='ouem-exploratory-crops-v2',purpose='PHASE_C3_RESEARCH_ONLY',samples=samples,
                target_ids=[s['image_id'] for s in candidates],candidate_manifest_sha256=candidate_sha,
                diagnosis_sha256='a'*64,metadata_sha256=candidate['metadata_sha256'])
    plan_path = root/'crop_plan.json'
    plan_path.write_text(json.dumps(plan))
    review = dict(schema='ouem-research-patch-review-v2',crop_plan_sha256=reader.r1.sha(plan_path),
                  formal_r1='UNCHANGED_HOLD',reviewer='SYNTHETIC',samples=reviews)
    review_path = root/'review.json'
    review_path.write_text(json.dumps(review))
    return root, plan, review_path, candidate_path


def open_pool(built):
    root, _, review, candidates = built
    return reader.ResearchData(root,reader.r1.sha(root/'crop_plan.json'),review,reader.r1.sha(review),
                               candidates=candidates,candidates_sha256=reader.r1.sha(candidates))


@pytest.mark.parametrize('count',[1,4,7])
def test_variable_source_balanced_schedule(pilot,tmp_path,count):
    built = pool(pilot,tmp_path,count)
    data = open_pool(built)
    plan = trainer.make_plan(data,steps=30)
    assert plan == trainer.make_plan(data,steps=30)
    other = trainer.make_plan(data,steps=30,seed=1)
    assert other['val'] == plan['val']
    if count > 1:
        assert other['train'] != plan['train']
    counts = Counter(s['image_id'] for s in plan['train'])
    assert max(counts.values())-min(counts.values()) <= 1
    train_count = sum(s['split']=='train' for s in data.samples.values())
    for start in range(0,30,train_count):
        cycle = plan['train'][start:start+train_count]
        assert len({s['image_id'] for s in cycle}) == len(cycle)
    for item in plan['train']+plan['val']:
        role = data.samples[item['image_id']]['split']
        image,target,products = data.patch(item,role)
        assert image.shape == (1,3,224,224) and target.shape == (1,224,224)
        assert torch.all((target==0)|(target==1)) and len(products)==3
        assert item['crop_xywh'] != [10,20,224,224] or item['image_id'] != 1054
    assert data.identity['candidate_manifest_sha256'] == reader.r1.sha(built[3])
    with pytest.raises(ValueError):
        data.patch(dict(image_id=1054,oam_id='SYNTHETIC-1054',crop_xywh=[10,20,224,224]),'train')


def test_pool_cpu_resume_and_identity(pilot,tmp_path):
    data = open_pool(pool(pilot,tmp_path,7))
    plan = trainer.make_plan(data,steps=2)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first = trainer.train(data,plan,Tiny,tmp_path/'segment',stop_after=1)
        final = trainer.train(data,plan,Tiny,tmp_path/'resume',resume=tmp_path/'segment/checkpoint.pth',
                              resume_sha256=first['checkpoint_sha256'])
    finally:
        torch.set_num_threads(old)
    assert final['completed_steps'] == 2 and final['history'][0] == first['history'][0]
    assert final['plan']['val'] == plan['val']
    state = trainer.safe_load(tmp_path/'resume/checkpoint.pth',final['checkpoint_sha256'])
    assert state['plan']['inputs'] == data.identity


@pytest.mark.parametrize('case',['candidate','alias','source','window','review','val','purpose'])
def test_pool_rejects_inconsistent_manifest(pilot,tmp_path,case):
    built = pool(pilot,tmp_path,5)
    root, plan, review_path, candidate_path = built
    review = json.loads(review_path.read_text())
    if case == 'candidate':
        plan['candidate_manifest_sha256'] = '0'*64
    elif case == 'alias':
        candidate = json.loads(candidate_path.read_text())
        candidate['candidates'][1]['oam_id'] = candidate['candidates'][0]['oam_id']
        candidate_path.write_text(json.dumps(candidate))
    elif case == 'source': plan['samples'][0]['source_hashes']['rgb'] = 'e'*64
    elif case == 'window': plan['samples'][0]['crop_xywh'][0] = 1
    elif case == 'review': review['samples'].pop()
    elif case == 'val':
        sample = copy.deepcopy(next(s for s in plan['samples'] if s['split']=='val'))
        plan['samples'].append(sample)
    else: plan['purpose'] = 'FORMAL_R1_PASS'
    path = root/'crop_plan.json'
    path.write_text(json.dumps(plan))
    review['crop_plan_sha256'] = reader.r1.sha(path)
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError): open_pool(built)


def test_generator_variable_targets_and_resample_once(inputs,tmp_path,monkeypatch):
    root,diagnosis,evidence = inputs
    rows = [json.loads(line) for line in (root/'archives/train_meta.json').read_text().splitlines()]
    ids = [rows[0]['image_id'],rows[-1]['image_id']]
    phase = []
    before = {}
    for row in rows:
        if row['image_id'] not in ids: continue
        row.update(height=1200,width=1200,bounds=[0,0,120,120])
        stem = f"{row['oam_id']}_{row['image_id']}"
        rgb = root/f'extracted/dataset/holdout/images/{stem}.tif'
        ann = root/f'extracted/dataset/holdout/masks/{stem}.png'
        with rasterio.open(rgb,'w',driver='GTiff',width=1200,height=1200,count=3,dtype='uint8',
                           crs='EPSG:3395',transform=rasterio.transform.from_origin(0,120,.1,.1)) as dst:
            dst.write(np.full((3,1200,1200),100,np.uint8))
        patches.Image.fromarray(np.zeros((1200,1200,3),np.uint8)).save(ann)
        shape,_,_ = patches.output_grid('EPSG:3395',rasterio.transform.from_origin(0,120,.1,.1),1200,1200)
        phase.append(dict(image_id=row['image_id'],oam_id=row['oam_id'],split='val' if row['validation_fold']==0 else 'train',
            technical='PASS',status='DIAGNOSED',global_teacher_fraction=0.,
            input_sha256={'rgb':patches.sha(rgb),'annotation':patches.sha(ann)},geometry={'output_shape':list(shape)},
            scenarios={'edge_black_only':{'patches':{'224':{'placement_count':(shape[0]-223)*(shape[1]-223)}}}}))
        before.update({rgb:patches.sha(rgb),ann:patches.sha(ann)})
    (root/'archives/train_meta.json').write_text('\n'.join(json.dumps(r) for r in rows))
    diagnosis.write_text(json.dumps({'images':phase}))
    manifest = patches.candidate_manifest(root,diagnosis,patches.sha(diagnosis))
    candidates = tmp_path/'candidates.json'
    candidates.write_text(json.dumps(manifest))
    calls = []
    original = patches.area_grid
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(patches,'area_grid',counted)
    result = patches.prepare(root,diagnosis,patches.sha(diagnosis),evidence,tmp_path/'generated',
        candidates_path=candidates,candidates_sha256=patches.sha(candidates),ids=ids,train_seeds=[0,1])
    assert len(calls)==2 and len(result['samples'])==3
    assert all(s['selected_black_support_cells']==0 for s in result['samples'])
    assert result['schema']=='ouem-exploratory-crops-v2'
    assert {p:patches.sha(p) for p in before} == before


@pytest.mark.parametrize('host',['oin-hotosm.s3.amazonaws.com','unapproved.invalid'])
def test_legacy_metadata_identifier_is_bound_not_fetched(pilot,tmp_path,host):
    built = pool(pilot,tmp_path,4)
    root, plan, review_path, candidate_path = built
    candidate = json.loads(candidate_path.read_text())
    reference = f'http://{host}/SYNTHETIC/object_meta.json'
    candidate['candidates'][0]['source_metadata'] = reference
    candidate_path.write_text(json.dumps(candidate))
    for sample in plan['samples']:
        if sample['image_id'] == 1054: sample['source_metadata'] = reference
    plan['candidate_manifest_sha256'] = reader.r1.sha(candidate_path)
    path = root/'crop_plan.json'
    path.write_text(json.dumps(plan))
    review = json.loads(review_path.read_text())
    review['crop_plan_sha256'] = reader.r1.sha(path)
    review_path.write_text(json.dumps(review))
    if host == 'oin-hotosm.s3.amazonaws.com':
        assert len(open_pool(built).samples)==4
    else:
        with pytest.raises(ValueError): open_pool(built)
