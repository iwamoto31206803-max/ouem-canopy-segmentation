import random
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import d6b_bundle as bundle
import d6b_relay as relay
import d6b_train as train
from research_val18_recovery import verified_load
from test_research_cuda import ToyModel


class StochasticToy(ToyModel):
    def forward(self,x,y):
        if self.training:
            x=x*(0.5+torch.rand(())+random.random()+float(np.random.rand()))
        return super().forward(x,y)


@pytest.fixture
def run(tmp_path):
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    samples={i:dict(image_id=i,split='train' if i<3 else 'val',oam_id=str(i),grid={'shape':[230,230]})
             for i in (1,2,3)}
    def patch(row,split):
        assert row['crop_xywh']==[2,3,224,224]
        assert row['image_id']<3 if split=='train' else row['image_id']>=3
        return (torch.full((1,3,224,224),.2*row['image_id']),
                torch.full((1,224,224),row['image_id']%2,dtype=torch.long),{})
    data=SimpleNamespace(samples=samples,crop_positions={i:[(2,3)] for i in samples},
                         identity={'dataset_id':'synthetic'},patch=patch)
    historical=SimpleNamespace(entries=[dict(image_id=4,oam_id='4',crop_xywh=[2,3,224,224])],
                               identity={'dataset_id':'old-synthetic'},patch=patch)
    binding,schedule=train.make_binding(dict(commit='1'*40,initializer_sha256='e'*64),'8'*64,data,historical,
                                        run_id='synthetic',total=4,save_every=2,eval_every=4)
    env={'software':{'torch':'test'},'device':'cpu'}

    def segment(name,updates,**kwargs):
        return train.run_segment(data,historical,kwargs.pop('binding',binding),schedule,StochasticToy,
                'cpu',env,name,tmp_path/('local-'+name),tmp_path/('relay-'+name),updates=updates,
                seconds=100,reserve=10,max_vram=1,**kwargs)
    try:
        yield segment,binding,schedule
    finally:
        torch.set_num_threads(previous)


def checkpoint(tmp_path,session,result):
    ref=[p for p in result['published'] if p['kind']=='checkpoint'][-1]
    return tmp_path/('relay-'+session)/ref['name'],ref['receipt_sha256']


def test_cold_relocated_round_trip_matches_continuous_cpu(run,tmp_path):
    segment,_,schedule=run
    continuous=segment('whole',4)
    first=segment('first',1)
    address,pin=checkpoint(tmp_path,'first',first)
    readback=tmp_path/'cold-readback'
    relay.copy(address,pin,readback)
    shutil.rmtree(tmp_path/'local-first')
    second=segment('second',3,resume=readback,receipt_sha=pin)
    assert second['parent']['completed']==1 and second['parent']['receipt_sha256']==pin
    assert second['status']==continuous['status']=='RUN_COMPLETE'
    logs=sorted(int(p.stem[-4:]) for name in ('first','second') for p in
                (tmp_path/('relay-'+name)).rglob('update-*.json'))
    assert logs==[r['step'] for r in schedule['train']]
    def state(name,result):
        folder,pin=checkpoint(tmp_path,name,result)
        ref=relay.verify(folder,pin)['files']['checkpoint.pth']
        return verified_load(folder/'checkpoint.pth',ref['sha256'],ref['size'],torch)
    assert state('second',second)['fingerprints']==state('whole',continuous)['fingerprints']
    assert state('second',second)['next_item'] is None
    final,final_pin=checkpoint(tmp_path,'second',second)
    init_pin=first['published'][0]['receipt_sha256']
    comp=train.compare_cohorts(tmp_path/'relay-first/evaluation-0000',init_pin,final,final_pin)
    assert comp['final_step']==4 and set(comp['cohorts'])=={'fresh_val25','historical_val18'}
    assert len(list(final.rglob('*.png')))==2


def test_resume_rejects_bad_hash_config_session_and_payload(run,tmp_path):
    segment,binding,_=run
    first=segment('first',2)
    address,pin=checkpoint(tmp_path,'first',first)
    with pytest.raises(ValueError,match='SHA|PIN'):
        segment('bad-pin',2,resume=address,receipt_sha='0'*64)
    with pytest.raises(ValueError,match='receipt binding'):
        segment('first',2,resume=address,receipt_sha=pin)
    with pytest.raises(ValueError,match='receipt binding'):
        segment('other',2,resume=address,receipt_sha=pin,binding=dict(binding,run_id='other'))
    (address/'checkpoint.pth').write_bytes(b'bad')
    with pytest.raises(ValueError,match='SHA'):
        segment('damaged',2,resume=address,receipt_sha=pin)


def test_expired_budget_avoids_updates(run):
    segment,_,_=run
    times=iter((0,95,95,95,95))
    result=segment('expired',2,clock=lambda:next(times))
    assert result['status']=='TIME_RESERVE' and result['completed']==0
    assert all(r['kind']!='checkpoint' for r in result['published'])


@pytest.mark.parametrize('names',[['../escape'],['a','A'],['C:/other']])
def test_archive_rejects_unsafe_members(names,tmp_path):
    archive=tmp_path/'bad.zip'
    with ZipFile(archive,'w') as z:
        for name in names:
            z.writestr(name,'x')
    with pytest.raises(ValueError,match='Unsafe'):
        bundle.unpack(archive,bundle.sha(archive),tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_interrupted_update_preserves_prior_relay(run,tmp_path,monkeypatch):
    segment,_,_=run
    original=train.d5.update
    count=0
    def update(*args):
        nonlocal count
        count+=1
        if count==3:
            raise RuntimeError('interrupted update')
        return original(*args)
    monkeypatch.setattr(train.d5,'update',update)
    with pytest.raises(RuntimeError,match='interrupted'):
        segment('interrupted',4)
    previous=tmp_path/'relay-interrupted/checkpoint-0002'
    pin=bundle.sha(previous/'receipt.json')
    assert relay.verify(previous,pin)['details']['completed']==2
    assert not (previous.parent/'checkpoint-0004').exists()
    monkeypatch.setattr(train.d5,'update',original)
    assert segment('recovered',2,resume=previous,receipt_sha=pin)['completed']==4
    assert bundle.sha(previous/'receipt.json')==pin
