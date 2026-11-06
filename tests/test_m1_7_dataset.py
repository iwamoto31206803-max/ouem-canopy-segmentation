"""Small offline contracts for frozen Zenodo10 verification; no training/data fetch."""
import copy
import json
from pathlib import Path
import shutil
import sys
import stat
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import m1_7_dataset as g0


@pytest.fixture
def frozen(tmp_path):
    ledger=json.loads((g0.REPO/'docs/implementation/M1_5B_PILOT_RESULT.json').read_text())
    samples=[];root=tmp_path/'dataset';root.mkdir()
    for stage in ['m1_5d2','m1_5d3']:
        samples.extend(copy.deepcopy(ledger[stage]['result']['samples']))
    for i,s in enumerate(samples):
        t=from_origin(i*100000,1000,.5,.5);folder=root/s['split']/str(s['image_id']);folder.mkdir(parents=True)
        s.update(status=g0.ACCEPTED,official_split='train',output_shape=[2,2],output_transform=list(t)[:6],
                 output_bounds=[i*100000,999,i*100000+1,1000],output_hashes={},
                 acceptance_evidence={k:'synthetic proof' for k in ['ledger_pointer','run_manifest_sha256','review_receipt_sha256','review_basis','review_preview_sha256','source_checks']})
        products={'rgb.tif':np.full((3,2,2),100,np.uint8),'canopy_fraction.tif':np.array([[[0,.5],[1,.25]]],np.float32),
                  'canopy_mask.tif':np.array([[[0,1],[1,0]]],np.uint8),'valid_fraction.tif':np.ones((1,2,2),np.float32)}
        for name,array in products.items():
            path=folder/name
            with rasterio.open(path,'w',driver='GTiff',width=2,height=2,count=array.shape[0],dtype=array.dtype,crs=s['output_crs'],transform=t) as dst:dst.write(array)
            s['output_hashes'][name]=g0.sha(path)
    manifest=dict(source_generation=g0.GENERATION,acceptance=g0.ACCEPTED,samples=samples)
    return root,manifest


def seal(root,manifest):
    manifest['dataset_id']=g0.identify(manifest)
    path=root/'manifest.json';path.write_bytes(g0.encoded(manifest))
    return path,g0.sha(path)


def test_frozen_copy_and_manifest_pin(frozen,tmp_path):
    root,m=frozen;path,digest=seal(root,m)
    first=g0.verify(root,path,digest)
    relocated=tmp_path/'other-volume';shutil.copytree(root,relocated)
    assert g0.verify(relocated,relocated/'manifest.json',digest)==first
    with pytest.raises(ValueError,match='SHA256'):g0.verify(root,path,'0'*64)
    m['scope']='modified';seal(root,m)
    with pytest.raises(ValueError,match='SHA256'):g0.verify(root,path,digest)


@pytest.mark.parametrize('change', ['id','split','hf','sample_hf','oam','holdout','fold','evidence','rights','qa'])
def test_frozen_admission_rejects(frozen,change):
    root,m=frozen;s=m['samples'][0]
    if change=='id':s['image_id']=999
    if change=='split':s['split']='val'
    if change=='hf':m['source_generation']='hf'
    if change=='sample_hf':s['source_generation']='hf'
    if change=='oam':s['oam_id']=m['samples'][1]['oam_id']
    if change=='holdout':s['official_split']='test'
    if change=='fold':s['validation_fold']=-1
    if change=='evidence':s['acceptance_evidence'].pop('review_receipt_sha256')
    if change=='rights':s['rights']['stac_sha256']=''
    if change=='qa':s.pop('validity_basis')
    path,digest=seal(root,m)
    with pytest.raises(ValueError):g0.verify(root,path,digest)


@pytest.mark.parametrize('change',['missing','bytes','extra','grid','label','invalid','fraction'])
def test_frozen_products_reject(frozen,change):
    root,m=frozen;s=m['samples'][0];folder=root/s['split']/str(s['image_id']);path=folder/'canopy_mask.tif'
    if change=='missing':path.unlink()
    elif change=='bytes':path.write_bytes(b'bad')
    elif change=='extra':shutil.copyfile(path,folder/'unexpected.tif')
    else:
        if change in ['invalid','fraction']:path=folder/('valid_fraction.tif' if change=='invalid' else 'canopy_fraction.tif')
        with rasterio.open(path,'r+') as dst:
            if change=='grid':dst.transform=from_origin(99,99,1,1)
            else:
                array=dst.read();array[0,0,0]=2 if change in ['label','fraction'] else 0;dst.write(array)
        s['output_hashes'][path.name]=g0.sha(path)  # ensure grid/label check, not merely a hash test
    manifest,digest=seal(root,m)
    with pytest.raises((ValueError,OSError)):g0.verify(root,manifest,digest)


def test_frozen_cross_role_overlap(frozen):
    root,m=frozen
    a=next(s for s in m['samples'] if s['split']=='train');b=next(s for s in m['samples'] if s['split']=='val')
    b['output_bounds']=a['output_bounds']
    path,digest=seal(root,m)
    with pytest.raises(ValueError,match='overlap'):g0.verify(root,path,digest)


@pytest.fixture
def transfer(frozen,tmp_path,monkeypatch):
    root,m=frozen;_,digest=seal(root,m)
    monkeypatch.setattr(g0,'MANIFEST_SHA256',digest);monkeypatch.setattr(g0,'DATASET_ID',m['dataset_id'])
    report=g0.pack(root,tmp_path/'bundle')
    return root,tmp_path/'bundle/zenodo10-g0.zip',report


def test_transfer_deterministic_pack_and_safe_unpack(transfer,tmp_path):
    root,archive,report=transfer
    assert report['file_count']==41 and report['dataset']['products']==40
    repeat=g0.pack(root,tmp_path/'repeat')
    assert repeat['zip_sha256']==report['zip_sha256']
    result=g0.unpack_report(archive,report['zip_sha256'],tmp_path/'colab',tmp_path/'result.json')
    assert result['status']=='PASS' and result['dataset']['dataset_id']==g0.DATASET_ID
    assert json.loads((tmp_path/'result.json').read_text())==result
    for item in report['files']:
        assert g0.sha(root/item['path'])==g0.sha(tmp_path/'colab'/item['path'])==item['sha256']
    with pytest.raises(ValueError):g0.pack(root,tmp_path/'bundle')


@pytest.mark.parametrize('attack',['traversal','absolute','backslash','drive','duplicate','case_duplicate','extra','missing','unexpected','symlink','directory','compression','manifest','bytes'])
def test_transfer_rejects_unsafe_archives(transfer,tmp_path,attack):
    _,archive,_=transfer
    with zipfile.ZipFile(archive) as src:
        members=[(info.filename,src.read(info)) for info in src.infolist()]
    name,data=members[-1]
    replacements={'traversal':'../escape.tif','absolute':'/escape.tif','backslash':'train\\escape.tif',
                  'drive':'C:/escape.tif','duplicate':members[0][0],'case_duplicate':members[0][0].upper(),
                  'unexpected':'notes.txt'}
    if attack in replacements:members[-1]=(replacements[attack],data)
    if attack=='extra':members.append(('extra.txt',b'extra'))
    if attack=='missing':members.pop()
    if attack=='manifest':members=[(n,b'{}' if n=='manifest.json' else d) for n,d in members]
    if attack=='bytes':members[-1]=(name,b'wrong')
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'x') as dest:
        for i,(n,d) in enumerate(members):
            info=zipfile.ZipInfo(n);info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16
            if i==len(members)-1:
                if attack=='symlink':info.external_attr=(stat.S_IFLNK|0o777)<<16
                if attack=='directory':info.external_attr=(stat.S_IFDIR|0o755)<<16
                if attack=='compression':info.compress_type=zipfile.ZIP_DEFLATED
            dest.writestr(info,d)
    with pytest.raises((ValueError,zipfile.BadZipFile)):g0.unpack(bad,g0.sha(bad),tmp_path/'out')
    assert not (tmp_path/'out').exists()
    assert not (tmp_path/'escape.tif').exists()


def test_transfer_pin_limits_and_overwrite_protection(transfer,tmp_path,monkeypatch):
    root,archive,report=transfer
    with pytest.raises(ValueError):g0.unpack(archive,'0'*64,tmp_path/'out')
    assert not (tmp_path/'out').exists()
    existing=tmp_path/'existing';existing.mkdir();(existing/'keep').write_text('protected')
    failed=g0.unpack_report(archive,report['zip_sha256'],existing,tmp_path/'failed.json')
    assert failed['status']=='FAIL' and (existing/'keep').read_text()=='protected'
    assert json.loads((tmp_path/'failed.json').read_text())['dataset_usable'] is False
    with pytest.raises(FileExistsError):g0.unpack_report(archive,report['zip_sha256'],tmp_path/'new',tmp_path/'failed.json')
    with pytest.raises(ValueError):g0.unpack_report(archive,report['zip_sha256'],root,root/'result.json')
    monkeypatch.setattr(g0,'MAX_TRANSFER_BYTES',1)
    with pytest.raises(ValueError,match='limit'):g0.unpack(archive,report['zip_sha256'],tmp_path/'large')


def test_pack_refuses_bad_input_before_creation(frozen,tmp_path,monkeypatch):
    root,m=frozen;_,digest=seal(root,m)
    monkeypatch.setattr(g0,'MANIFEST_SHA256',digest)
    (root/m['samples'][0]['split']/str(m['samples'][0]['image_id'])/'rgb.tif').unlink()
    with pytest.raises(OSError):g0.pack(root,tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_transfer_partial_output_is_never_a_pass(transfer,tmp_path,monkeypatch):
    _,archive,report=transfer
    def fail_after_copy(*args,**kwargs): raise ValueError('simulated final grid verification failure')
    monkeypatch.setattr(g0,'verify',fail_after_copy)
    output=tmp_path/'partial'
    result=g0.unpack_report(archive,report['zip_sha256'],output,tmp_path/'partial-result.json')
    assert result['status']=='FAIL' and result['dataset_usable'] is False
    assert output.exists()  # preserved for diagnosis; never silently deleted/reused
    assert json.loads((tmp_path/'partial-result.json').read_text())['status']=='FAIL'
