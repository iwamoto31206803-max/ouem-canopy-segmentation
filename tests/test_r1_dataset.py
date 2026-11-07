"""Offline generic R1 contracts; synthetic rasters, never real source acquisition."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import r1_dataset as r


def seal(m):
    m['dataset_id'] = r.identify(m)
    return m


@pytest.fixture
def dataset(tmp_path):
    m = dict(schema=r.SCHEMA, source_generation=r.GENERATION, samples=[])
    for i in range(3):
        h, w = i+2, i+3
        split = 'val' if i == 2 else 'train'
        t = from_origin(i*100000, 10000, .5, .5)
        source = dict(dataset='restor/tcd Zenodo', revision=r.GENERATION, original_source_id='scene-'+str(i),
                      image_sha256=r.hashlib.sha256(f'image-{i}'.encode()).hexdigest(), annotation_sha256='b'*64)
        source['footprint_wgs84'] = list(r.transform_bounds('EPSG:3395',4326,i*100000,10000-h*.5,i*100000+w*.5,10000))
        review = dict(status='PASS', evidence_ref='synthetic receipt', evidence_sha256='c'*64,
                      source_hashes={k: source[k] for k in ('image_sha256', 'annotation_sha256')})
        s = dict(image_id=i, oam_id='oam-'+str(i), split=split, official_split='train', validation_fold=0 if i==2 else 1,
                 source_generation=r.GENERATION, threshold=.5, source=source,
                 grid=dict(crs='EPSG:3395', transform=list(t)[:6], bounds=[i*100000,10000-h*.5,i*100000+w*.5,10000], shape=[h,w]),
                 rights=dict(image_license='CC-BY-4.0', annotation_license='CC-BY-4.0', original_source_license='CC-BY-4.0',
                             attribution='synthetic provider', evidence_ref='synthetic STAC receipt', evidence_sha256='d'*64),
                 labels=dict(policy='upstream_nonzero_binary_with_review', categories=[1,2]), products={})
        for key in ('category_review','validity_review','r1_review','independence_review'): s[key]=copy.deepcopy(review)
        s['external_independence_review']=dict(copy.deepcopy(review),
            scope='official_holdout_and_unlisted_neighbor_sources',catalog_sha256='e'*64)
        fraction = np.full((1,h,w), i/2, np.float32)
        arrays = {'rgb.tif':np.full((3,h,w), 120, np.uint8), 'canopy_fraction.tif':fraction,
                  'canopy_mask.tif':(fraction>=.5).astype(np.uint8), 'valid_fraction.tif':np.ones((1,h,w),np.float32)}
        for name, a in arrays.items():
            relative=f'{split}/{i}/{name}'; path=tmp_path/relative; path.parent.mkdir(parents=True,exist_ok=True)
            with rasterio.open(path,'w',driver='GTiff',height=h,width=w,count=a.shape[0],dtype=a.dtype,crs='EPSG:3395',transform=t) as dst: dst.write(a)
            s['products'][name]=dict(path=relative,sha256=r.sha(path))
        m['samples'].append(s)
    return tmp_path,seal(m)


def rewrite(root,s,name,a=None,**profile):
    p=root/s['products'][name]['path']
    with rasterio.open(p) as src: old=src.read(); meta=src.profile
    meta.update(profile)
    with rasterio.open(p,'w',**meta) as dst: dst.write(old if a is None else a)
    s['products'][name]['sha256']=r.sha(p)


def test_counts_sizes_uniform_threshold_copy_loader(dataset,tmp_path):
    root,m=dataset
    assert r.verify(root,m)['acceptance']=='PASS'
    verified=r.preflight(root,m)
    for s in m['samples']:
        arrays=r.load_sample(verified,s['image_id'],s['split'])
        assert np.array_equal(arrays['canopy_mask.tif'],(arrays['canopy_fraction.tif']>=.5))
    one=copy.deepcopy(m);one['samples']=one['samples'][:1]
    assert r.verify(root,seal(one))['technical']=='PASS'
    target=tmp_path/'copy';shutil.copytree(root,target,ignore=shutil.ignore_patterns('copy'))
    assert r.verify(root,m)==r.verify(target,m)
    reverse=copy.deepcopy(m);reverse['samples'].reverse()
    assert r.identify(m)==r.identify(reverse)
    with pytest.raises(ValueError,match='split'):r.load_sample(verified,2,'train')


@pytest.mark.parametrize('change',[
    lambda m:m.pop('schema'), lambda m:m.update(samples=[]),
    lambda m:m['samples'].append(copy.deepcopy(m['samples'][0])),
    lambda m:m['samples'][0].update(official_split='test'),
    lambda m:m['samples'][0].update(validation_fold=0),
    lambda m:m['samples'][0].update(source_generation='HF'),
    lambda m:m['samples'][0]['grid'].update(crs='invalid'),
    lambda m:m['samples'][0]['products']['rgb.tif'].update(path='../rgb.tif'),
    lambda m:m['samples'][0]['products']['rgb.tif'].update(sha256='bad'),
])
def test_invalid_manifest(dataset,change):
    root,m=dataset;change(m);seal(m)
    assert r.verify(root,m)['technical']=='FAIL'


@pytest.mark.parametrize('what',['hash','missing','grid','crs','dtype','pixel','threshold','partial','range','label','nodata','channels'])
def test_bad_products(dataset,what):
    root,m=dataset;s=m['samples'][0];p=root/s['products']['rgb.tif']['path']
    if what=='hash':p.write_bytes(b'corrupt')
    elif what=='missing':p.unlink()
    elif what=='grid':rewrite(root,s,'rgb.tif',transform=from_origin(0,20000,.5,.5))
    elif what=='crs':rewrite(root,s,'rgb.tif',crs='EPSG:4326')
    elif what=='dtype':rewrite(root,s,'rgb.tif',dtype='float32')
    elif what=='pixel':rewrite(root,s,'canopy_fraction.tif',np.full((1,2,3),np.nan,np.float32))
    elif what=='threshold':rewrite(root,s,'canopy_mask.tif',np.ones((1,2,3),np.uint8))
    elif what=='partial':rewrite(root,s,'valid_fraction.tif',np.zeros((1,2,3),np.float32))
    elif what=='range':rewrite(root,s,'canopy_fraction.tif',np.full((1,2,3),1.1,np.float32))
    elif what=='label':rewrite(root,s,'canopy_mask.tif',np.full((1,2,3),2,np.uint8))
    elif what=='nodata':rewrite(root,s,'rgb.tif',nodata=0)
    elif what=='channels':rewrite(root,s,'rgb.tif',np.ones((1,2,3),np.uint8),count=1)
    report=r.verify(root,seal(m))
    assert report['technical']==('HOLD' if what in ('partial','nodata') else 'FAIL')
    assert report['acceptance']!='PASS'


@pytest.mark.parametrize('change',[
    lambda s:s.pop('rights'),lambda s:s['rights'].update(original_source_license='CC-BY-NC-4.0'),
    lambda s:s.pop('category_review'),lambda s:s['labels'].update(categories=[3]),
    lambda s:s['source'].pop('original_source_id'),lambda s:s['r1_review'].update(status='accepted_for_M1_7_smoke'),
    lambda s:s['validity_review']['source_hashes'].update(image_sha256='f'*64),
    lambda s:s['source'].update(revision='HF-revision'),
    lambda s:s['source'].update(footprint_wgs84=[-10,-10,-9,-9]),
])
def test_missing_or_conflicting_evidence_holds(dataset,change):
    root,m=dataset;change(m['samples'][0]);report=r.verify(root,seal(m))
    assert report['technical']=='PASS' and report['acceptance']=='HOLD'


@pytest.mark.parametrize('what',['same_oam','same_scene','same_bytes','overlap','near','same_role'])
def test_independence(dataset,what):
    root,m=dataset;a=m['samples'][0];b=m['samples'][2]
    if what=='same_oam':b['oam_id']=a['oam_id']
    elif what=='same_scene':b['source']['original_source_id']=a['source']['original_source_id']
    elif what=='same_bytes':b['source']['image_sha256']=a['source']['image_sha256']
    elif what=='same_role':m['samples'][1]['oam_id']=a['oam_id']
    else:
        b['grid']['bounds']=a['grid']['bounds'] if what=='overlap' else [10,9999,11,10000]
    issues=r.independence(m['samples'])
    assert any(issues.values())
    if what in ('same_oam','same_scene','same_bytes'):assert r.verify(root,seal(m))['acceptance']=='FAIL'


def test_cli_json_failures_and_overwrite(dataset,tmp_path):
    root,m=dataset;p=tmp_path/'manifest.json';p.write_text(json.dumps(m));report=tmp_path/'result.json'
    args=['--root',str(root),'--manifest',str(p),'--manifest-sha256',r.sha(p),'--report',str(report)]
    assert r.main(args)==0
    with pytest.raises(FileExistsError):r.main(args)
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='Duplicate'):r.read_manifest(p)
    args[-1]=str(tmp_path/'failed.json')
    assert r.main(args)==1
    assert json.loads(Path(args[-1]).read_text())['technical']=='FAIL'


def test_legacy_projection_never_promotes():
    old=r.read_manifest(Path(__file__).parents[1]/'docs/implementation/M1_7_G0_MANIFEST.json')
    new=r.from_g0(old);r.contract(new)
    assert len(new['samples'])==10
    assert all(r.admission(s) for s in new['samples'])


def test_symlink_rejected(tmp_path):
    target=tmp_path/'real';target.write_text('x');link=tmp_path/'link'
    try:link.symlink_to(target)
    except OSError:pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(ValueError,match='Symlink'):r.safe_path(tmp_path,'link')


def test_reparse_guard_without_windows_privilege(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path=tmp_path/'file';path.write_text('x');original=Path.lstat
    def lstat(p,*args,**kwargs):
        if p==path:return SimpleNamespace(st_mode=r.stat.S_IFREG,st_file_attributes=1024)
        return original(p,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',lstat)
    with pytest.raises(ValueError,match='reparse'):r.safe_path(tmp_path,'file')


@pytest.mark.parametrize('value',[[],None,{'schema':r.SCHEMA,'source_generation':r.GENERATION,'samples':[None]}])
def test_non_object_manifest_reports_fail(tmp_path,value):
    assert r.verify(tmp_path,value)['technical']=='FAIL'


def test_identity_content_and_missing_category_binding(dataset):
    root,m=dataset
    original=m['dataset_id'];m['samples'][0]['rights']['attribution']='different provider'
    assert r.identify(m)!=original
    assert r.verify(root,m)['technical']=='FAIL'
    seal(m);assert r.verify(root,m)['acceptance']=='PASS'
    m['samples'][0]['category_review'].pop('source_hashes')
    assert r.verify(root,seal(m))['acceptance']=='HOLD'


@pytest.mark.parametrize('problem',['unselected_hash','unselected_hold','unselected_partial'])
def test_loading_requires_dataset_wide_preflight(dataset,problem):
    root,m=dataset;other=m['samples'][1]
    if problem=='unselected_hash':(root/other['products']['rgb.tif']['path']).write_bytes(b'corrupt')
    elif problem=='unselected_hold':other.pop('category_review')
    else:rewrite(root,other,'valid_fraction.tif',np.zeros((1,3,4),np.float32))
    seal(m)
    with pytest.raises(ValueError,match='preflight'):r.preflight(root,m)
    with pytest.raises(ValueError,match='preflight'):r.load_sample(r.verify(root,m),0,'train')


def test_preflight_snapshot_binding_and_selected_mutation(dataset):
    root,m=dataset;verified=r.preflight(root,m)
    with pytest.raises(ValueError,match='preflight'):r.load_sample(None,0,'train')
    with pytest.raises(ValueError,match='preflight'):r.load_sample(r._VerifiedDataset(root,b'{}',None),0,'train')
    m['samples'][0]['split']='val'
    assert r.load_sample(verified,0,'train')['rgb.tif'].shape==(3,2,3)
    (root/'train/0/rgb.tif').write_bytes(b'changed after preflight')
    with pytest.raises(ValueError,match='SHA256'):r.load_sample(verified,0,'train')


@pytest.mark.parametrize('problem',['hold','late_fail','threshold'])
def test_product_accounting_preserves_partial_progress(dataset,problem):
    root,m=dataset;s=m['samples'][0]
    if problem=='hold':rewrite(root,s,'valid_fraction.tif',np.zeros((1,2,3),np.float32))
    elif problem=='late_fail':(root/s['products']['rgb.tif']['path']).write_bytes(b'corrupt')
    else:rewrite(root,s,'canopy_mask.tif',np.ones((1,2,3),np.uint8))
    result=r.verify(root,seal(m));counts=result['product_counts']
    assert 'products_verified' not in result
    assert counts['expected']==12
    assert counts['attempted']+counts['not_attempted']==12
    assert counts['pass']+counts['hold']+counts['fail']==counts['attempted']
    if problem=='hold':assert counts==dict(expected=12,attempted=12,not_attempted=0,**{'pass':11,'hold':1,'fail':0})
    elif problem=='late_fail':assert counts==dict(expected=12,attempted=11,not_attempted=1,**{'pass':10,'hold':0,'fail':1})
    else:
        assert counts['pass']==12 and result['technical']=='FAIL'
        assert result['samples_fully_technical_pass']==2


@pytest.mark.parametrize('problem',['missing','scope','catalog','binding'])
def test_external_independence_is_separate_from_internal_checks(dataset,problem):
    root,m=dataset;s=m['samples'][0]
    if problem=='missing':s.pop('external_independence_review')
    elif problem=='scope':s['external_independence_review']['scope']='manifest_only'
    elif problem=='catalog':s['external_independence_review'].pop('catalog_sha256')
    else:s['external_independence_review']['source_hashes']['image_sha256']='f'*64
    report=r.verify(root,seal(m))
    assert report['technical']=='PASS' and report['acceptance']=='HOLD'
    assert report['independence']['within_manifest']['status']=='PASS'
    assert report['independence']['external_sources']['status']=='NOT_VERIFIED'
    assert not report['independence']['external_sources']['review_assertions_complete']
    with pytest.raises(ValueError,match='preflight'):r.preflight(root,m)


def test_external_assertions_are_not_offline_proof(dataset):
    root,m=dataset;report=r.verify(root,m)
    assert report['acceptance']=='PASS'
    assert report['independence']['external_sources']['review_assertions_complete']
    assert report['independence']['external_sources']['status']=='NOT_VERIFIED'
