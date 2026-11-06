"""CPU-only G2 schedule, crop, transaction and tiny-model continuation contracts."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import m1_7_g2_train as g2
from test_m1_7_dataset import frozen,seal


def manifest():return json.loads((g2.data.REPO/'docs/implementation/M1_7_G0_MANIFEST.json').read_text())


def test_schedule_counts_seed_order_and_bounds():
    m=manifest();a=g2.make_schedule(m)
    assert a==g2.make_schedule(m,0) and a!=g2.make_schedule(m,1)
    counts=g2.Counter(s['image_id'] for s in a)
    assert set(counts)==set(g2.TRAIN_IDS) and sorted(counts.values())==[2,3,3,3,3,3,3]
    assert all(len({s['image_id'] for s in a[start:start+7]})==len(a[start:start+7]) for start in [0,7,14])
    assert g2.config(0)['steps']==20
    for bad in [-1,True,2**32]:
        with pytest.raises(ValueError):g2.config(bad)


@pytest.mark.parametrize('case',['val','holdout','negative','outside','size','duplicate','length','oam'])
def test_invalid_schedule_rejected(case):
    m=manifest();s=g2.make_schedule(m)
    if case=='val':s[0]['image_id']=1972
    if case=='holdout':s[0]['image_id']=99999
    if case=='negative':s[0]['crop_xywh'][0]=-1
    if case=='outside':s[0]['crop_xywh'][0]=99999
    if case=='size':s[0]['crop_xywh'][2]=223
    if case=='duplicate':s[1].update(image_id=s[0]['image_id'],oam_id=s[0]['oam_id'],crop_xywh=s[0]['crop_xywh'])
    if case=='length':s.pop()
    if case=='oam':s[0]['oam_id']='wrong'
    with pytest.raises(ValueError):g2.validate_schedule(s,m)


def checkpoint_metadata():
    m=manifest();s=g2.make_schedule(m)
    return dict(kind=g2.KIND,config=g2.config(0),inputs=g2.inputs(m),initializer={'sha256':'initial'},completed_steps=10,next_step=11,
                executed_steps=list(range(1,11)),remaining_steps=list(range(11,21)),schedule=s,schedule_sha256=g2.digest(s),
                history=[dict(item,step_loss=.5) for item in s[:10]],environment={'torch':'test'},git_sha='commit')


@pytest.mark.parametrize('case',['kind','config','data','initializer','count','next','remaining','schedule','history','nan','executed'])
def test_checkpoint_compatibility_rejects(case):
    state=checkpoint_metadata();base=copy.deepcopy(state)
    if case=='kind':state['kind']=g2.shared.KIND
    if case=='config':state['config']['lr']=.1
    if case=='data':state['inputs']['source_generation']='HF'
    if case=='initializer':state['initializer']={}
    if case=='count':state['completed_steps']=11
    if case=='next':state['next_step']=10
    if case=='remaining':state['remaining_steps']=[]
    if case=='schedule':state['schedule_sha256']='wrong'
    if case=='history':state['history'][0]['image_id']=1972
    if case=='nan':state['history'][0]['step_loss']=float('nan')
    if case=='executed':state['executed_steps']=[]
    with pytest.raises(ValueError):g2.compatible(state,manifest(),base['config'],base['initializer'],base['environment'],'commit',True)


def test_checkpoint_environment_and_claim(tmp_path):
    s=checkpoint_metadata()
    assert g2.compatible(s,manifest(),s['config'],s['initializer'],s['environment'],'commit')=={}
    with pytest.raises(ValueError):g2.compatible(s,manifest(),s['config'],s['initializer'],{'torch':'new'},'commit')
    assert g2.compatible(s,manifest(),s['config'],s['initializer'],{'torch':'new'},'commit',True)
    g2.claim_continuation(tmp_path,'a'*64)
    with pytest.raises(FileExistsError):g2.claim_continuation(tmp_path,'a'*64)
    with pytest.raises(ValueError):g2.claim_continuation(tmp_path,'../unsafe')


def test_nonzero_crop_alignment_no_padding(frozen,monkeypatch):
    root,m=frozen
    for i,s in enumerate(m['samples']):
        t=from_origin(i*100000,1000,.5,.5);s.update(output_shape=[4,5],output_transform=list(t)[:6],output_bounds=[i*100000,998,i*100000+2.5,1000])
        values=np.arange(20,dtype=np.uint8).reshape(4,5);mask=(values%2).astype(np.uint8)
        arrays={'rgb.tif':np.repeat(values[None],3,axis=0),'canopy_mask.tif':mask[None],
                'canopy_fraction.tif':mask[None].astype(np.float32),'valid_fraction.tif':np.ones((1,4,5),np.float32)}
        for name,a in arrays.items():
            path=root/s['split']/str(s['image_id'])/name
            with rasterio.open(path,'w',driver='GTiff',height=4,width=5,count=a.shape[0],dtype=a.dtype,crs=s['output_crs'],transform=t) as dst:dst.write(a)
            s['output_hashes'][name]=g2.data.sha(path)
    path,h=seal(root,m);monkeypatch.setattr(g2.data,'MANIFEST_SHA256',h)
    monkeypatch.setattr(g2.shared,'CONFIG',dict(g2.shared.CONFIG,patch=2))
    x,y,crops,meta=g2.shared.load_patch(root,path,3,'train',[1,1,2,2])
    assert np.array_equal(crops['rgb.tif'][0],np.array([[6,7],[11,12]]))
    assert np.array_equal(y.numpy()[0],np.array([[0,1],[1,0]]))
    assert torch.equal(x[0],torch.from_numpy(crops['rgb.tif']).float()/255)
    assert meta['crop_xywh']==[1,1,2,2]
    for crop in [[-1,0,2,2],[4,0,2,2],[0,0,3,2]]:
        with pytest.raises(ValueError):g2.shared.load_patch(root,path,3,'train',crop)
    for value in [0,1]:
        s=next(s for s in m['samples'] if s['image_id']==3)
        for name in ['canopy_mask.tif','canopy_fraction.tif']:
            p=root/'train/3'/name
            with rasterio.open(p,'r+') as dst:dst.write(np.full((1,4,5),value,dtype=dst.dtypes[0]))
            s['output_hashes'][name]=g2.data.sha(p)
        path,h=seal(root,m);monkeypatch.setattr(g2.data,'MANIFEST_SHA256',h)
        assert torch.all(g2.shared.load_patch(root,path,3,'train',[1,1,2,2])[1]==value)


def arguments(tmp_path,output='run',backup='drive',*extra):
    return ['--data',str(tmp_path/'data'),'--checkpoint',str(tmp_path/'initial.pth'),'--g1a-result',str(tmp_path/'g1a.json'),
            '--run-id','cpu-pilot','--output',str(tmp_path/output),'--backup',str(tmp_path/backup),*extra]


def test_limits_and_existing_output(tmp_path):
    for value in ['0','21','100']:
        with pytest.raises(SystemExit):g2.parser().parse_args(arguments(tmp_path)+['--stop-after',value])
    args=g2.parser().parse_args(arguments(tmp_path));g2.reserve(args)
    with pytest.raises(ValueError):g2.reserve(args)
    with pytest.raises(ValueError):g2.reserve(g2.parser().parse_args(arguments(tmp_path,'restore')+['--restore-only']))


@pytest.fixture
def cpu_gpu_surrogate(tmp_path,monkeypatch):
    """No official model, CUDA allocation or real image training occurs in this fixture."""
    import satlaspretrain_models as satlas
    torch.set_num_threads(4)
    (tmp_path/'data').mkdir();(tmp_path/'data/manifest.json').write_bytes(g2.data.encoded(manifest()))
    torch.save({'tiny':torch.ones(1)},tmp_path/'initial.pth');initial={'sha256':g2.data.sha(tmp_path/'initial.pth')}
    class Tiny(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__();self.drop=torch.nn.Dropout(.2)
            for name,n in [('backbone',3),('fpn',2),('upsample',2),('head',2)]:setattr(self,name,torch.nn.Conv2d(n,2,1))
        def to(self,**kwargs):return self
        def forward(self,x,y):
            features=self.drop(self.backbone(x))*(1+random.random()*.01)
            logits=self.head(self.upsample(self.fpn(features)))
            return logits.softmax(1),torch.nn.functional.cross_entropy(logits,y)
    monkeypatch.setattr(satlas,'Model',Tiny)
    original_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        return original_to(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    rng=[torch.tensor([1,2],dtype=torch.uint8)]
    for name,fn in dict(is_available=lambda:False,manual_seed_all=lambda seed:None,reset_peak_memory_stats=lambda:None,
        synchronize=lambda:None,memory_allocated=lambda:0,max_memory_allocated=lambda:0,memory_reserved=lambda:0,max_memory_reserved=lambda:0,
        get_rng_state_all=lambda:rng,set_rng_state_all=lambda states:None,device_count=lambda:1).items():monkeypatch.setattr(torch.cuda,name,fn)
    def preflight(args,record):record.update(status='PREFLIGHT_PASS_CPU_EMULATION_ONLY',initializer=initial,environment={'torch':'CPU_SURROGATE'},git_sha='test-commit')
    monkeypatch.setattr(g2.shared,'run',preflight)
    def patch(root,path,image_id,role,crop):
        assert role=='train' and image_id in g2.TRAIN_IDS
        x=torch.full((1,3,224,224),.1+(image_id%17+crop[0]+crop[1])/512)
        y=((torch.arange(224)[:,None]+torch.arange(224))%2).long()[None]
        return x,y,{},dict(image_id=image_id,crop_xywh=crop,positive_fraction=.5)
    monkeypatch.setattr(g2.shared,'load_patch',patch)
    return tmp_path


def test_cpu_surrogate_step10_resume_matches_continuous20(cpu_gpu_surrogate,monkeypatch):
    tmp=cpu_gpu_surrogate
    assert g2.main(arguments(tmp,'continuous','drive-a'))==0
    continuous=json.loads((tmp/'continuous/result.json').read_text())
    assert continuous['completed_steps']==20 and sorted(continuous['scene_counts'].values())==[2,3,3,3,3,3,3]
    assert g2.main(arguments(tmp,'paused','drive-b')+['--stop-after','10'])==0
    paused=json.loads((tmp/'paused/result.json').read_text());receipt=paused['checkpoints'][0]
    path=tmp/'drive-b/step10'/receipt['filename']
    with pytest.raises(ValueError):g2.shared.safe_load(path,'0'*64)
    original_schedule=g2.make_schedule
    monkeypatch.setattr(g2,'make_schedule',lambda *a,**kw:pytest.fail('resume must not redraw schedule'))
    resume=arguments(tmp,'resumed','drive-b')+['--resume',str(path),'--resume-sha256',receipt['sha256']]
    assert g2.main(resume)==0
    result=json.loads((tmp/'resumed/result.json').read_text())
    assert result['schedule']==paused['schedule'] and result['executed_steps']==list(range(1,21))
    assert result['checkpoints'][-1]['fingerprints']==continuous['checkpoints'][-1]['fingerprints']
    assert g2.main(arguments(tmp,'duplicate','drive-b')+['--resume',str(path),'--resume-sha256',receipt['sha256']])==1
    duplicate=json.loads((tmp/'duplicate/result.json').read_text())
    assert duplicate['completed_steps']==10 and duplicate['status']=='FAIL_STOPPED'
    monkeypatch.setattr(g2,'make_schedule',original_schedule)
    last=result['checkpoints'][-1]
    assert g2.main(arguments(tmp,'restore','drive-b')+['--resume',str(tmp/'drive-b/step20'/last['filename']),
        '--resume-sha256',last['sha256'],'--restore-only'])==0
    assert all(v=='PASS' for v in json.loads((tmp/'restore/result.json').read_text())['restore_checks'].values())


def test_cpu_surrogate_backup_failure_preserves_logs_and_checkpoint(cpu_gpu_surrogate,monkeypatch):
    tmp=cpu_gpu_surrogate
    def failure(*a,**kw):raise OSError('simulated backup failure')
    monkeypatch.setattr(g2.shared,'copy_backup',failure)
    assert g2.main(arguments(tmp))==1
    result=json.loads((tmp/'run/result.json').read_text())
    assert result['status']=='FAIL_STOPPED' and result['completed_steps']==10
    assert len(result['history'])==10 and (tmp/'run/zenodo10_g2_step10.pth').exists()
    assert not list((tmp/'drive').glob('*.claim.json')) and not (tmp/'run/zenodo10_g2_step20.pth').exists()


def test_nonfinite_failure_is_recorded_without_progress(cpu_gpu_surrogate,monkeypatch):
    tmp=cpu_gpu_surrogate
    def fail(*args,**kwargs):
        args[5].update(stage='backward',nan_inf_detected=False)
        raise ValueError('Nonfinite gradient in synthetic component')
    monkeypatch.setattr(g2.shared,'one_step',fail)
    assert g2.main(arguments(tmp))==1
    result=json.loads((tmp/'run/result.json').read_text())
    assert result['status']=='FAIL_STOPPED' and result['completed_steps']==0 and result['nan_inf_detected'] is True
    assert result['active_step']['nan_inf_detected'] is True and result['active_step']['stage']=='backward'
