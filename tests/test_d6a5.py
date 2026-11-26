from pathlib import Path

import pytest

from test_d6a32_gate import fake_real, synthetic
import d6a32_gate as gate
import d6a3_guard
import d6a3_produce as p
from d6a4 import D6Shards, SOURCE_FIELDS, aggregate_document
from d6a5_inventory import HOST, RESPONSE_LIMIT, collect, response
from r2_train import make_plan


@pytest.mark.parametrize('synthetic',[(320,8)],indirect=True)
def test_composed_dataset_pins_old_generation(fake_real, monkeypatch):
    plan,write,(base,root,evidence,out),_=fake_real
    monkeypatch.setattr(d6a3_guard,'allocated_bytes',lambda p:p.stat().st_size)
    inputs=[{k:r[k] for k in SOURCE_FIELDS} for r in plan['sources']]
    pair=plan['a1']['train']

    def publish(plan,name):
        link=write('plan.json',plan)
        ready=gate.verify_evidence(plan,link['sha256'],base,root,evidence)
        path=out.with_name(name)
        info=p.publish(base/'plan.json',link['sha256'],root,path,ready,real=True)
        return dict(path=str(path),sha256=info['manifest_sha256'])

    pilot=publish(dict(plan,sources=inputs[:4]),'pilot')
    old=dict(schema=gate.ROSTER,pilot_sha256=pilot['sha256'],sources=inputs[:6],shard_sizes=[2])
    old_link=write('old-roster.json',old)
    a4=publish(dict(plan,schema=gate.SHARD,a1=[pair],roster=old_link,shard_start=4,sources=inputs[4:6]),'a4')
    parent=write('parent.json',aggregate_document(old_link,old,[pilot,a4],6))
    parent['path']=str(base/parent['path'])
    roster=dict(schema=gate.EXTENDED_ROSTER,parent_count=6,parent_sha256=parent['sha256'],
                sources=inputs,shard_sizes=[2])
    link=write('roster.json',roster)
    plan_new=dict(plan,schema=gate.SHARD,a1=[pair],roster=link,shard_start=6,sources=inputs[6:])
    shard=publish(plan_new,'a5')

    def read(shards, parent_ref=parent, total=8):
        doc=aggregate_document(link,roster,shards,total,parent=parent_ref)
        dest=write('aggregate.json',doc)
        return D6Shards(base/dest['path'],dest['sha256'])

    data=read([shard])
    assert data.complete and len(data.samples)==8
    schedule=make_plan(data,steps=4)
    for split in ('train','val'):
        assert len(schedule[split])==4
        for row in schedule[split]:
            assert data.patch(row,split)[0].shape==(1,3,224,224)
            with pytest.raises(ValueError,match='D6_PATCH_IDENTITY'):
                data.patch(row,'val' if split=='train' else 'train')
    assert not read([],total=6).complete
    with pytest.raises(ValueError,match='D6_PARENT_PIN'):
        read([shard],dict(parent,sha256='0'*64))
    with pytest.raises(ValueError,match='Missing'):
        read([dict(shard,path=str(out/'missing'))])
    with pytest.raises(ValueError,match='D6_SHARD_ORDER_OR_ROSTER'):
        read([a4])
    with pytest.raises(ValueError,match='D6_ROSTER_DUPLICATE'):
        gate.roster_sources(dict(roster,sources=inputs[:7]+[inputs[0]]))
    Path(parent['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='INPUT_PIN_MISMATCH'):
        read([shard])


def test_lookup_stops_at_rate_limit_and_never_retries(tmp_path):
    order=[dict(oam_id=f'{i:024x}') for i in range(5)]
    calls=[]
    def fetch(oid):
        calls.append(oid)
        return (404 if len(calls)==1 else 429),'application/json',b'','HTTP_ERROR'
    log=tmp_path/'log.jsonl'
    collect(order,tmp_path,log,5,fetch=fetch,delay=0)
    assert calls==[r['oam_id'] for r in order[:2]]
    assert len(log.read_bytes().splitlines())==2
    assert not list(tmp_path.glob('*-stac.json'))
    with pytest.raises(FileExistsError):
        collect(order,tmp_path,log,5,fetch=fetch,delay=0)


def test_transport_uses_one_official_get_and_bounds_read():
    calls=[]
    class Reply:
        status=200
        def getheader(self,name,default=None):
            return {'Content-Type':'application/geo+json'}.get(name,default)
        def read(self,n):
            assert n==RESPONSE_LIMIT+1
            return b'{}'
    class Connection:
        def __init__(self,host,timeout):
            assert host==HOST and timeout==12
        def request(self,method,path,headers):
            calls.append((method,path))
            assert headers['Accept-Encoding']=='identity'
        def getresponse(self):
            return Reply()
        def close(self):
            calls.append('closed')
    oid='1'*24
    assert response(oid,Connection)==(200,'application/geo+json',b'{}',None)
    assert calls==[('GET','/stac/collections/openaerialmap/items/'+oid),'closed']
    with pytest.raises(ValueError):
        response('../other-host',Connection)
