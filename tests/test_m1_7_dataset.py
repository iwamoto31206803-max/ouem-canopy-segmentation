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


def test_zenodo_gpu_loader_alignment_roles_and_uniform(frozen,tmp_path,monkeypatch):
    import torch
    import m1_7_g1b_gpu_smoke as gpu
    root,m=frozen;manifest,digest=seal(root,m)
    monkeypatch.setattr(g0,'MANIFEST_SHA256',digest)
    monkeypatch.setattr(gpu,'CONFIG',dict(gpu.CONFIG,patch=2))  # tiny synthetic IO contract, not real 224 inference
    first=gpu.load_patch(root,manifest,3,'train');again=gpu.load_patch(root,manifest,3,'train')
    x,y,crops,meta=first
    assert x.shape==(1,3,2,2) and x.dtype==torch.float32 and y.dtype==torch.long
    assert torch.equal(x,again[0]) and torch.equal(y,torch.from_numpy(crops['canopy_mask.tif']).long())
    assert meta==again[3] and meta['crop_xywh']==[0,0,2,2]
    assert torch.equal(x[0],torch.from_numpy(crops['rgb.tif']).float()/255)
    for image_id,role in [(1972,'train'),(3,'val'),(999,'train')]:
        with pytest.raises(ValueError):gpu.load_patch(root,manifest,image_id,role)
    for value in [0,1]:
        s=next(s for s in m['samples'] if s['image_id']==3)
        for name in ['canopy_fraction.tif','canopy_mask.tif']:
            path=root/'train/3'/name
            with rasterio.open(path,'r+') as dst:dst.write(np.full((1,2,2),value,dtype=dst.dtypes[0]))
            s['output_hashes'][name]=g0.sha(path)
        manifest,digest=seal(root,m);monkeypatch.setattr(g0,'MANIFEST_SHA256',digest)
        assert torch.all(gpu.load_patch(root,manifest,3,'train')[1]==value)
    monkeypatch.setattr(g0,'MANIFEST_SHA256','0'*64)
    with pytest.raises(ValueError):gpu.load_patch(root,manifest,3,'train')


def test_zenodo_gpu_receipt_gate(tmp_path):
    import m1_7_g1b_gpu_smoke as gpu
    report=json.loads((g0.REPO/'docs/implementation/M1_7_G1A_TRANSFER.json').read_text())['windows_unpack_validation']
    path=tmp_path/'g1a.json';path.write_text(json.dumps(report))
    assert gpu.g1a_receipt(path)['status']=='PASS'
    report['dataset']['dataset_id']='HF';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):gpu.g1a_receipt(path)


def test_zenodo_gpu_cli_protection_and_missing_cuda(tmp_path,monkeypatch):
    import torch
    import m1_7_g1b_gpu_smoke as gpu
    args=['--data',str(tmp_path/'data'),'--checkpoint',str(tmp_path/'official.pth'),
          '--g1a-result',str(tmp_path/'g1a.json'),'--output',str(tmp_path/'run'),'--preflight-only']
    with pytest.raises(SystemExit):gpu.parser().parse_args(args+['--image-id','1972'])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert gpu.main(args)==2
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['status']=='CUDA_NOT_VERIFIED' and result['completed_steps']==0
    assert result['forward_seconds']=='NOT_MEASURED'
    with pytest.raises(SystemExit):gpu.main(args)  # prior result is protected
    with pytest.raises(ValueError):gpu.reserve(gpu.parser().parse_args(args[:-1]))  # backup required for step


def test_zenodo_gpu_tiny_cpu_step_state_and_backup(tmp_path,monkeypatch):
    """CPU surrogate only; no real Satlas construction or CUDA success evidence."""
    import random
    import torch
    import m1_7_g1b_gpu_smoke as gpu
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name,channels in [('backbone',3),('fpn',2),('upsample',2),('head',2)]:setattr(self,name,torch.nn.Conv2d(channels,2,1))
        def forward(self,x,y):
            logits=self.head(self.upsample(self.fpn(self.backbone(x))))
            return logits.softmax(1),torch.nn.functional.cross_entropy(logits,y)
    torch.set_num_threads(2);torch.manual_seed(0);random.seed(0)
    for name in ['synchronize']:monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ['memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved']:monkeypatch.setattr(torch.cuda,name,lambda:0)
    rng=[torch.tensor([1,2],dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:rng)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda states:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    record={};images=torch.rand(1,3,224,224);target=torch.zeros(1,224,224,dtype=torch.long)
    gpu.one_step(model,optimizer,images,target,torch,record)
    assert record['completed_steps']==1 and all(s['changed'] for s in record['updates'].values())
    assert all(g['finite_nonzero_parameters']>0 for g in record['gradients'].values())
    state=dict(kind=gpu.KIND,config=gpu.CONFIG,completed_steps=1,model=model.state_dict(),optimizer=optimizer.state_dict(),
               cpu_rng=torch.get_rng_state(),cuda_rng=rng,python_rng=random.getstate(),fingerprints=gpu.fingerprints(model,optimizer,torch),
               inputs={'dataset_id':g0.DATASET_ID,'source_generation':g0.GENERATION,'image_id':3},initializer={'sha256':'fixed'},environment={'torch':'example'},git_sha='commit')
    path=tmp_path/'tiny.pth';digest=gpu.save_state(path,state)
    payload=gpu.safe_load(path,digest)
    assert gpu.compatible(payload,state['inputs'],state['initializer'],state['environment'],'commit')=={}
    restored=Tiny();opt=torch.optim.AdamW(restored.parameters())
    assert all(v=='PASS' for v in gpu.restore(restored,opt,payload,torch).values())
    for key,value in [('kind','HF'),('inputs',{}),('initializer',{}),('completed_steps',2),('config',{})]:
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError):gpu.compatible(bad,state['inputs'],state['initializer'],state['environment'],'commit',True)
    with pytest.raises(ValueError):gpu.compatible(payload,state['inputs'],state['initializer'],{'torch':'new'},'commit')
    assert gpu.compatible(payload,state['inputs'],state['initializer'],{'torch':'new'},'commit',True)
    for component in payload['fingerprints']:
        bad=copy.deepcopy(payload);bad['fingerprints'][component]='wrong'
        with pytest.raises(ValueError,match='fingerprint'):gpu.restore(restored,opt,bad,torch)
    with pytest.raises(ValueError):gpu.safe_load(path,'0'*64)
    backup=gpu.copy_backup(path,digest,tmp_path/'drive')
    assert backup['copy_sha_match'] and g0.sha(tmp_path/'drive/tiny.pth')==digest
    with pytest.raises(ValueError):gpu.copy_backup(path,digest,tmp_path/'drive')
    with pytest.raises(FileExistsError):gpu.save_state(path,state)


def test_zenodo_gpu_cpu_surrogate_full_cli_and_restore(tmp_path,monkeypatch):
    """CPU emulation of orchestration ONLY, not a measured GPU/Satlas run."""
    from types import SimpleNamespace
    import torch
    import satlaspretrain_models as satlas
    import m1_1_satlas_smoke as shared
    import m1_7_g1b_gpu_smoke as gpu
    class Tiny(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            for name,channels in [('backbone',3),('fpn',2),('upsample',2),('head',2)]:setattr(self,name,torch.nn.Conv2d(channels,2,1))
        def to(self,**kwargs):return self
        def forward(self,x,y):
            logits=self.head(self.upsample(self.fpn(self.backbone(x))))
            return logits.softmax(1),torch.nn.functional.cross_entropy(logits,y)
    monkeypatch.setattr(satlas,'Model',Tiny)
    monkeypatch.setattr(shared,'checkpoint_digest',lambda *a:None)
    original_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        return original_to(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    monkeypatch.setattr(gpu.subprocess,'check_output',lambda cmd,**kw:'' if cmd[1]=='status' else 'cpu-emulated-commit\n')
    rng=[torch.tensor([1,2],dtype=torch.uint8)]
    for name,fn in dict(is_available=lambda:True,get_device_properties=lambda i:SimpleNamespace(name='CPU_TEST_ONLY',total_memory=0),
        device_count=lambda:1,set_device=lambda i:None,manual_seed_all=lambda s:None,reset_peak_memory_stats=lambda:None,
        synchronize=lambda:None,memory_allocated=lambda:0,max_memory_allocated=lambda:0,memory_reserved=lambda:0,max_memory_reserved=lambda:0,
        get_rng_state_all=lambda:rng,set_rng_state_all=lambda states:None).items():monkeypatch.setattr(torch.cuda,name,fn)
    images=torch.ones(1,3,224,224);target=torch.zeros(1,224,224,dtype=torch.long)
    inputs=dict(dataset_id=g0.DATASET_ID,manifest_sha256=g0.MANIFEST_SHA256,source_generation=g0.GENERATION,image_id=3)
    monkeypatch.setattr(gpu,'load_patch',lambda *a:(images.clone(),target.clone(),{},dict(inputs)))
    monkeypatch.setattr(gpu,'g1a_receipt',lambda *a:{'status':'CPU_EMULATED_ONLY'})
    initial=tmp_path/'initial.pth';torch.save({'tiny':torch.ones(1)},initial)
    common=['--data',str(tmp_path/'data'),'--checkpoint',str(initial),'--g1a-result',str(tmp_path/'g1a.json')]
    assert gpu.main(common+['--output',str(tmp_path/'step'),'--backup',str(tmp_path/'drive')])==0
    first=json.loads((tmp_path/'step/result.json').read_text())
    assert first['completed_steps']==1 and first['backup_checkpoint']['copy_sha_match']
    assert gpu.main(common+['--output',str(tmp_path/'restored'),'--restore',str(tmp_path/'drive/zenodo10_step1.pth'),
                            '--restore-sha256',first['saved_checkpoint']['sha256']])==0
    restored=json.loads((tmp_path/'restored/result.json').read_text())
    assert all(v=='PASS' for v in restored['restore_checks'].values())
    assert restored['optimizer_seconds']=='NOT_MEASURED'  # restore never takes another optimizer step


def test_zenodo_gpu_failure_stops_without_fallback(tmp_path,monkeypatch):
    import m1_7_g1b_gpu_smoke as gpu
    def oom(args,record):record['stage']='optimizer_step';raise RuntimeError('simulated CUDA out of memory')
    monkeypatch.setattr(gpu,'run',oom)
    args=['--data',str(tmp_path/'data'),'--checkpoint',str(tmp_path/'official.pth'),'--g1a-result',str(tmp_path/'g1a.json'),
          '--output',str(tmp_path/'failed'),'--backup',str(tmp_path/'drive')]
    assert gpu.main(args)==1
    result=json.loads((tmp_path/'failed/result.json').read_text())
    assert result['status']=='FAIL_STOPPED' and result['stage']=='optimizer_step'
    assert not (tmp_path/'drive').exists() and not (tmp_path/'failed/zenodo10_step1.pth').exists()
