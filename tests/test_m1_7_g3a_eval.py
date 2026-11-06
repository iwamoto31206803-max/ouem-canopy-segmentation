"""CPU-only metric boundaries and inference/artifact contracts; no real Swin/checkpoint."""
import copy
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import m1_7_g3a_eval as ev
from test_m1_7_dataset import frozen,seal
from test_m1_7_g2_train import checkpoint_metadata,manifest


@pytest.mark.parametrize('pred,truth,expected',[
    ([[1,0]],[[1,0]],(1,1,1,1)),
    ([[1,0]],[[0,0]],(0,0,0,None)),
    ([[0,0]],[[1,0]],(0,0,None,0)),
    ([[0,0]],[[0,0]],(None,None,None,None)),
    ([[1,1]],[[1,1]],(1,1,1,1)),
])
def test_metric_boundaries(pred,truth,expected):
    result=ev.confusion(np.array(pred),np.array(truth))
    assert tuple(result[k] for k in ['tree_iou','f1','precision','recall'])==expected
    assert result['dice']==result['f1']
    assert all(k in result['undefined_reasons'] for k in ['tree_iou','f1','precision','recall'] if result[k] is None)
    assert result['tn']+result['tp']+result['fp']+result['fn']==2


def test_macro_micro_ce_and_pp():
    a=dict(ev.count_metrics(1,0,0,1),image_id=1972,ce=.2)
    b=dict(ev.count_metrics(0,3,1,0),image_id=269,ce=.8)
    agg=ev.aggregate([a,b])
    assert agg['macro']['metrics']['tree_iou']==.5
    assert agg['micro']['tree_iou']==.2
    assert agg['ce_valid_pixel_mean']==pytest.approx((.2*2+.8*4)/6)
    empty=dict(ev.count_metrics(0,0,0,2),image_id=269,ce=.1)
    agg=ev.aggregate([a,empty])
    assert agg['macro']['valid_image_counts']['precision']==1
    assert ev.aggregate([empty])['macro']['metrics']['f1'] is None
    other=dict(a,tree_iou=.75,ce=.3)
    delta=ev.comparison([a],[other])['delta_b_minus_a']
    assert delta['per_image'][0]['percentage_points']['tree_iou']==-25
    assert delta['ce_valid_pixel_mean']==pytest.approx(.1)
    assert ev.delta(empty,a)['percentage_points']['precision'] is None


def test_signed_area_and_model_differences():
    a=dict(ev.count_metrics(0,0,2,2),image_id=1972,ce=1.)
    b=dict(ev.count_metrics(2,0,0,2),image_id=1972,ce=.2)
    assert a['area_error']==-.5 and a['absolute_area_error']==.5
    assert ev.count_metrics(0,2,0,2)['area_error']==.5
    d=ev.comparison([a],[b])['delta_b_minus_a']
    assert d['macro']['percentage_points']['tree_iou']==100
    assert d['micro']['percentage_points']['absolute_area_error']==-50
    assert d['micro']['count_difference']['tp']==2 and d['ce_valid_pixel_mean']==pytest.approx(-.8)


class Tiny(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__();self.head=torch.nn.Module();self.head.layers=torch.nn.Conv2d(3,2,1)
        self.modes=[]
    def to(self,*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return super().to(*args,**kwargs)
    def forward(self,x,y):
        self.modes.append((self.training,torch.is_grad_enabled(),torch.is_inference_mode_enabled()))
        logits=self.head.layers(x)
        return logits.softmax(1),torch.nn.functional.cross_entropy(logits,y)


def patches():
    return {rid:(torch.ones(1,3,224,224),torch.zeros(1,224,224,dtype=torch.long),
                 {'rgb.tif':np.full((3,224,224),120,np.uint8)},dict(split='val',crop_xywh=[0,0,224,224])) for rid in ev.VAL_IDS}


def test_logits_ce_modes_and_immutable_model():
    torch.set_num_threads(2);model=Tiny()
    with torch.no_grad():model.head.layers.weight.zero_();model.head.layers.bias.copy_(torch.tensor([0.,1.]))
    before={k:v.clone() for k,v in model.state_dict().items()}
    rows,arrays,checks=ev.evaluate_model(model,patches(),'cpu')
    assert checks['unchanged'] and len(rows)==3 and all(m==(False,False,True) for m in model.modes)
    assert rows[0]['ce']==pytest.approx(float(torch.log1p(torch.exp(torch.tensor(1.)))),rel=1e-6)
    assert arrays[1972]['probability'][0,0]==pytest.approx(.7310586)
    assert rows[0]['fp']==224**2 and rows[0]['recall'] is None
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    tie=Tiny()
    with torch.no_grad():tie.head.layers.weight.zero_();tie.head.layers.bias.zero_()
    tied,_,_=ev.evaluate_model(tie,patches(),'cpu')
    assert tied[0]['tn']==224**2 and tied[0]['tree_iou'] is None
    assert tied[0]['ce']==pytest.approx(float(np.log(2)),rel=1e-6)
    with pytest.raises(ValueError):ev.evaluate_model(model,{3:next(iter(patches().values()))},'cpu')


def test_mismatched_size_and_state_mutation_stop():
    class Mutating(Tiny):
        def __init__(self):super().__init__();self.register_buffer('counter',torch.zeros(1))
        def forward(self,x,y):self.counter.add_(1);return super().forward(x,y)
    with pytest.raises(ValueError,match='changed'):ev.evaluate_model(Mutating(),patches(),'cpu')
    class Wrong(Tiny):
        def __init__(self):super().__init__();self.head.layers=torch.nn.Conv2d(3,2,1,stride=2)
        def forward(self,x,y):
            z=self.head.layers(x);return z.softmax(1),z.sum()*0
    with pytest.raises(ValueError,match='shape'):ev.evaluate_model(Wrong(),patches(),'cpu')


def metadata20():
    from m1_4_bounded_overfit import fingerprint
    s=checkpoint_metadata();s.update(completed_steps=20,next_step=21,executed_steps=list(range(1,21)),remaining_steps=[],
        history=[dict(item,step_loss=.5) for item in s['schedule']],model={'tiny':torch.ones(1)},fingerprints={'model':'test'})
    s['initializer']=json.loads((ev.data.REPO/'docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    s['environment']['satlaspretrain-models']='0.3.1'
    s['fingerprints']['model']=fingerprint(s['model'])
    return s


@pytest.mark.parametrize('key,value',[('kind','G1B'),('completed_steps',10),('config',{'architecture':'other'}),('inputs',{'source_generation':'HF'})])
def test_bad_checkpoint_metadata(key,value):
    s=metadata20();s[key]=value
    with pytest.raises(ValueError):ev.validate_checkpoint(s,manifest(),metadata20()['initializer'])


def test_hash_and_shape_crop(frozen,monkeypatch):
    root,m=frozen
    for i,s in enumerate(m['samples']):
        t=from_origin(i*100000,1000,.5,.5);s.update(output_shape=[230,231],output_transform=list(t)[:6],output_bounds=[i*100000,885,i*100000+115.5,1000])
        vals=(np.indices((230,231)).sum(0)%2).astype(np.uint8)
        arrays={'rgb.tif':np.repeat((vals*200)[None],3,axis=0),'canopy_mask.tif':vals[None],
                'canopy_fraction.tif':vals[None].astype(np.float32),'valid_fraction.tif':np.ones((1,230,231),np.float32)}
        for name,a in arrays.items():
            p=root/s['split']/str(s['image_id'])/name
            with rasterio.open(p,'w',driver='GTiff',height=230,width=231,count=a.shape[0],dtype=a.dtype,crs=s['output_crs'],transform=t) as dst:dst.write(a)
            s['output_hashes'][name]=ev.data.sha(p)
    _,h=seal(root,m);monkeypatch.setattr(ev.data,'MANIFEST_SHA256',h)
    _,p=ev.load_val(root)
    assert list(p)==ev.VAL_IDS
    assert p[1972][0].shape==(1,3,224,224) and p[1972][3]['excluded_pixels']==230*231-224**2
    assert np.array_equal(p[1972][1][0].numpy(),p[1972][2]['canopy_mask.tif'][0])
    checkpoint=root/'fake.pth';torch.save({'x':torch.ones(1)},checkpoint)
    with pytest.raises(ValueError):ev.shared.safe_load(checkpoint,'0'*64)


def test_render_publish_and_failure_protection(tmp_path):
    root=tmp_path/'out';root.mkdir();p=patches();a={'prediction':np.zeros((224,224),np.uint8),'probability':np.full((224,224),.3,np.float32)}
    original=p[1972][2]['rgb.tif'].copy()
    for rid in ev.VAL_IDS:ev.render(root/f'val_{rid}.png',p[rid][2]['rgb.tif'].transpose(1,2,0),np.zeros((224,224),np.uint8),a,a)
    assert Image.open(root/'val_1972.png').size==(960,512)
    assert np.array_equal(original,p[1972][2]['rgb.tif'])
    ev.write_new(root/'metrics.json',{'test':'synthetic'});ev.write_new(root/'provenance.json',{'dataset_id':ev.data.DATASET_ID})
    result={'status':'NOT_VERIFIED'};ev.publish(root,tmp_path/'drive',result)
    assert result['status']=='G3A_EVALUATION_AND_COPY_PASS'
    hashes=json.loads((root/'sha256.json').read_text())['sha256']
    assert len(hashes)==6
    for name,h in hashes.items():assert ev.data.sha(root/name)==ev.data.sha(tmp_path/'drive'/name)==h
    with pytest.raises(ValueError):ev.publish(root,tmp_path/'drive',result)


def test_missing_cuda_failure_never_passes(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    args=['--data',str(tmp_path/'data'),'--initializer',str(tmp_path/'init.pth'),'--trained',str(tmp_path/'trained.pth'),
          '--output',str(tmp_path/'run'),'--preflight-only']
    assert ev.main(args)==1
    assert json.loads((tmp_path/'run/result.json').read_text())['status']=='FAIL_STOPPED'
    with pytest.raises(SystemExit):ev.main(args)
    lineage=tmp_path/'old-g2';lineage.mkdir();(lineage/'lineage.json').write_text('{}')
    parsed=ev.parser().parse_args(args);parsed.output=lineage/'new-child'
    with pytest.raises(ValueError,match='lineages'):ev.reserve(parsed)
    assert not parsed.output.exists()


def test_reconstructed_seed_and_exact_component_checks(tmp_path,monkeypatch):
    import satlaspretrain_models as satlas
    import satlaspretrain_models.model as official
    weights={f'{prefix}.{name}':tensor for prefix in ['b','f'] for name,tensor in torch.nn.Linear(2,2).state_dict().items()}
    path=tmp_path/'initial.pth';torch.save(weights,path);monkeypatch.setattr(ev,'INITIAL_SHA256',ev.data.sha(path))
    class Small(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__();self.backbone=torch.nn.Linear(2,2);self.fpn=torch.nn.Linear(2,2)
            self.upsample=torch.nn.Linear(2,2);self.head=torch.nn.Linear(2,2)
            self.backbone.load_state_dict({k:kwargs['weights']['b.'+k] for k in ['weight','bias']})
            self.fpn.load_state_dict({k:kwargs['weights']['f.'+k] for k in ['weight','bias']})
    calls=[]
    def adjust(w,name,prefix,count):
        calls.append((name,prefix,count));key='b' if name=='backbone' else 'f'
        return {k:w[key+'.'+k] for k in ['weight','bias']}
    monkeypatch.setattr(satlas,'Model',Small);monkeypatch.setattr(official,'adjust_state_dict_prefix',adjust)
    torch.manual_seed(123);a,checks=ev.reconstructed(path,'cpu')
    torch.manual_seed(456);b,_=ev.reconstructed(path,'cpu')
    assert all(torch.equal(v,b.head.state_dict()[k]) for k,v in a.head.state_dict().items())
    assert checks['fpn']['status']=='EXACT_OFFICIAL_TENSORS'
    assert calls[:2]==[('backbone','backbone.',1),('fpn','intermediates.0.',0)]
    def wrong(w,name,prefix,count):return {k:v+1 for k,v in adjust(w,name,prefix,count).items()}
    monkeypatch.setattr(official,'adjust_state_dict_prefix',wrong)
    with pytest.raises(ValueError,match='weights differ'):ev.reconstructed(path,'cpu')


def test_cpu_surrogate_full_cli_provenance_and_copy(tmp_path,monkeypatch):
    import satlaspretrain_models as satlas
    import satlaspretrain_models.model as official_model
    import satlaspretrain_models.models.heads as official_heads
    from m1_4_bounded_overfit import fingerprint
    initial=tmp_path/'initial.pth';torch.save({'tiny':torch.ones(1)},initial)
    state=metadata20();model=Tiny();state['model']=model.state_dict();state['fingerprints']['model']=fingerprint(model.state_dict())
    state['environment']['official_source_sha256']={n:ev.data.sha(Path(inspect.getfile(m))) for n,m in [('model',official_model),('head',official_heads)]}
    trained=tmp_path/'trained.pth';torch.save(state,trained)
    monkeypatch.setattr(ev,'INITIAL_SHA256',ev.data.sha(initial));monkeypatch.setattr(ev,'TRAINED_SHA256',ev.data.sha(trained))
    monkeypatch.setattr(satlas,'Model',Tiny)
    monkeypatch.setattr(ev,'reconstructed',lambda weights,device:(Tiny(),{'scope':'CPU_EMULATION_ONLY'}))
    monkeypatch.setattr(ev,'load_val',lambda root:(manifest(),patches()))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_device',lambda i:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(name='CPU_TEST_NOT_GPU',total_memory=0))
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(ev.subprocess,'check_output',lambda cmd,**kw:'' if cmd[1]=='status' else 'cpu-test-commit\n')
    original_to=torch.Tensor.to
    def to_cpu(tensor,*a,**kw):
        if a and a[0]=='cuda':a=('cpu',*a[1:])
        return original_to(tensor,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to_cpu)
    args=['--data',str(tmp_path/'data'),'--initializer',str(initial),'--trained',str(trained),
          '--output',str(tmp_path/'run'),'--backup',str(tmp_path/'drive')]
    assert ev.main(args)==0
    provenance=json.loads((tmp_path/'run/provenance.json').read_text())
    assert provenance['model_a']['label']=='RECONSTRUCTED_STEP0'
    assert provenance['model_b']['evaluation_state']['unchanged']
    assert json.loads((tmp_path/'run/result.json').read_text())['gpu_evaluation']=='COMPLETED'
    assert ev.data.sha(initial)==ev.INITIAL_SHA256 and ev.data.sha(trained)==ev.TRAINED_SHA256
    wrong=copy.deepcopy(state)
    wrong['model']['head.layers.weight']=torch.zeros(3,3,1,1);wrong['model']['head.layers.bias']=torch.zeros(3)
    wrong['fingerprints']['model']=fingerprint(wrong['model'])
    bad=tmp_path/'wrong-structure.pth';torch.save(wrong,bad)
    monkeypatch.setattr(ev,'TRAINED_SHA256',ev.data.sha(bad))
    bad_args=list(args)
    for old,new in [(trained,bad),(tmp_path/'run',tmp_path/'wrong-run'),(tmp_path/'drive',tmp_path/'wrong-drive')]:bad_args[bad_args.index(str(old))]=str(new)
    assert ev.main(bad_args)==1
    assert json.loads((tmp_path/'wrong-run/result.json').read_text())['stage']=='model_b'
    monkeypatch.setattr(ev,'TRAINED_SHA256',ev.data.sha(trained))
    original_copy=ev.shutil.copyfileobj
    def broken(*args,**kw):raise OSError('simulated copy failure')
    monkeypatch.setattr(ev.shutil,'copyfileobj',broken)
    args[args.index(str(tmp_path/'run'))]=str(tmp_path/'failed')
    args[args.index(str(tmp_path/'drive'))]=str(tmp_path/'failed-drive')
    assert ev.main(args)==1
    assert json.loads((tmp_path/'failed/result.json').read_text())['status']=='FAIL_STOPPED'
    calls=[]
    def late_failure(*a,**kw):
        calls.append(1)
        if len(calls)==7:raise OSError('simulated checksum-list copy failure')
        return original_copy(*a,**kw)
    monkeypatch.setattr(ev.shutil,'copyfileobj',late_failure)
    args[args.index(str(tmp_path/'failed'))]=str(tmp_path/'late')
    args[args.index(str(tmp_path/'failed-drive'))]=str(tmp_path/'late-drive')
    assert ev.main(args)==1
    assert json.loads((tmp_path/'late/result.json').read_text())['status']=='FAIL_STOPPED'
    assert json.loads((tmp_path/'late-drive/FAILURE.json').read_text())['status']=='FAIL_STOPPED'
