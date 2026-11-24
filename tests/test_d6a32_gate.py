import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import d6a32_gate as gate
import d6a3_produce as producer
from test_d6a3_produce import synthetic


@pytest.fixture
def contract(tmp_path):
    plan = dict(schema=gate.PLAN, source_generation=gate.qa.GENERATION, gt54=gate.GT54.copy(),
                windows_per_source={'train': 1, 'val': 1},
                limits=dict(gate.MAXIMUMS, **gate.MINIMUMS), allow_authenticity=sorted(gate.AUTH),
                sources=[dict(image_id=1, split='train'), dict(image_id=2, split='val')])
    def files(change=lambda _: None):
        p, a = tmp_path/'plan.json', tmp_path/'approved.json'
        p.write_bytes(producer.canonical(plan))
        pin=gate.qa.sha(p)
        token=dict(schema=gate.APPROVAL,approved=True,plan_sha256=pin,purpose='ONE_BOUNDED_REAL_OUTPUT_PILOT')
        change(token)
        a.write_bytes(producer.canonical(token))
        return p,pin,a,gate.qa.sha(a)
    return plan, files


@pytest.mark.parametrize('invalid',['pin','false','numeric','binding','cohort','gt54','budget'])
def test_real_approval_fails_closed(contract,invalid):
    plan,files=contract
    if invalid=='cohort':
        plan['sources'].extend([dict(image_id=i,split='val') for i in (3,4)])
    if invalid=='gt54':
        plan['gt54']=['PASS']
    if invalid=='budget':
        plan['limits']['physical_bytes']+=1
    changes={'false':dict(approved=False),'numeric':dict(approved=1),'binding':dict(plan_sha256='0'*64)}
    p,pin,a,ap=files(lambda t:t.update(changes.get(invalid,{})))
    with pytest.raises(ValueError):
        gate.approval(p,pin,a,'0'*64 if invalid=='pin' else ap)


def test_valid_metadata_still_requires_physical_quota(contract,tmp_path,monkeypatch):
    _,files=contract
    p,pin,a,ap=files()
    monkeypatch.setattr(gate,'verify_evidence',lambda *_:None)
    def forbidden(*_a,**_k):
        pytest.fail('Real producer was reached')
    monkeypatch.setattr(producer,'produce',forbidden)
    output=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['producer','--plan',str(p),'--plan-sha256',pin,
        '--root',str(tmp_path),'--evidence',str(tmp_path),'--output',str(output),
        '--real-approval',str(a),'--real-approval-sha256',ap])
    with pytest.raises(SystemExit) as exc:
        producer.main()
    assert exc.value.code==1 and not output.exists()


@pytest.mark.parametrize('role,prior_role,match,box,expected',[
    ('train','train',True,[0,0,.01,.01],True),
    ('train','train',False,[0,0,.01,.01],False),
    ('train','val',True,[0,0,.01,.01],None),
    ('val','train',True,[0,0,.01,.01],None),
    ('val','val',True,[0,0,.01,.01],None),
    ('train','val',False,[0,0,.01,.01],None),
    ('val','train',False,[0,0,.01,.01],None),
    ('val','train',False,[1,1,1.01,1.01],False),
])
def test_known_protection_and_same_role_reuse(role,prior_role,match,box,expected):
    row=dict(image_id=1,oam_id='scene',split=role,official_split='train',
             validation_fold=1 if role=='train' else 0,license='CC-BY 4.0',meta='https://example.com/scene')
    group=dict(wgs84_bounds=[0,0,.01,.01],center=[.005,.005])
    old=dict(oam_id='scene' if match else 'other',alias=gate.qa.identity(row) if match else '0'*64,
             split=prior_role,envelope=box)
    if expected is None:
        with pytest.raises(ValueError):
            gate.protected_lineage(row,group,[old])
    else:
        assert gate.protected_lineage(row,group,[old]) is expected


def test_relative_reference_cannot_escape(tmp_path):
    plan={'protected':dict(path='../escape.json',sha256='a'*64)}
    with pytest.raises(ValueError,match='Unsafe'):
        gate.verify_evidence(plan,tmp_path,tmp_path,tmp_path)


def test_fake_real_2048_metadata_and_a1_crop_binding(synthetic, monkeypatch):
    _, _, old, _, base = synthetic
    root=base/'cache'
    paths={k:root/v for k,v in gate.qa.FILES.items()}
    rows,groups,objects=gate.qa.load_catalog(paths)
    prior=[]
    for i in range(88):
        rid=i+100
        oid=f'prior-{rid}'
        role='train' if i<70 else 'val'
        row=dict(image_id=rid,oam_id=oid,split=role,official_split='train',validation_fold=0 if role=='val' else 1,
                 meta=f'https://metadata.example/{oid}')
        box=[10,10,11,11]
        rows[rid]=row
        groups[oid]=dict(row,wgs84_bounds=box,center=[10.5,10.5],roles={role})
        prior.append(dict(image_id=rid,oam_id=oid,split=role,alias=gate.qa.identity(row),envelope=box))
    rows[1].update(width=2048,height=2048)
    monkeypatch.setattr(gate.qa,'load_catalog',lambda _: (rows,groups,objects))
    refs={k:old[k] for k in ('a1_contract','a1_report')}
    def write(name,data):
        path=base/name
        path.write_bytes(producer.canonical(data))
        return dict(path=name,sha256=gate.qa.sha(path))
    freeze=write('freeze.json',dict(identity=dict(crop_plan_sha256='1'*64)))
    cat={k:gate.qa.sha(path) for k,path in paths.items()}
    protected=write('protected.json',dict(scope='v5 and nested historical Val18 only', sources=prior,
        provenance=dict(freeze_sha256=freeze['sha256'],crop_plan_sha256='1'*64,
                        metadata_sha256={f'{k}_meta.json':cat[k] for k in ('train','test')})))
    records=json.loads((base/'report.json').read_bytes())['results']
    plan=dict(a1={role:dict(contract=refs['a1_contract'],report=refs['a1_report']) for role in ('train','val')},
        protected=protected,historical_freeze=freeze,catalog_sha256=cat,
        allow_authenticity=sorted(gate.AUTH),
        sources=[dict(x,historical_train_reuse=False) for x in records])
    gate.verify_evidence(plan,base,root,base/'stac')
    plan['sources'][0]['crop_xywh']=[0,0,224,224]
    with pytest.raises(ValueError,match='REAL_A1_SOURCE_BINDING'):
        gate.verify_evidence(plan,base,root,base/'stac')
