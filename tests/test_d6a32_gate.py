import copy
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
                windows_per_source={'train': 1, 'val': 1}, storage_policy=gate.STORAGE_POLICY,
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


def test_cli_approval_requires_both_flags(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['producer','--plan',str(tmp_path/'plan.json'),'--plan-sha256','0'*64,
        '--root',str(tmp_path),'--evidence',str(tmp_path),'--output',str(tmp_path/'out'),
        '--real-approval',str(tmp_path/'approval.json')])
    with pytest.raises(SystemExit) as exc:
        producer.main()
    assert exc.value.code==1 and not (tmp_path/'out').exists()


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
    plan={'protected':dict(path='../escape.json',sha256=gate.SUBINDEX_SHA),
          'historical_freeze':dict(sha256=gate.FREEZE_SHA)}
    with pytest.raises(ValueError,match='Unsafe'):
        gate.verify_evidence(plan,'0'*64,tmp_path,tmp_path,tmp_path)


@pytest.fixture
def fake_real(synthetic, monkeypatch):
    _, out, _, _, base = synthetic
    root, evidence = base/'cache', base/'stac'
    paths={k:root/v for k,v in gate.qa.FILES.items()}
    with paths['train'].open('ab') as stream:
        for i in range(88):
            oid=f'prior-{i}'
            fold=1 if i<70 else 0
            stream.write(producer.canonical(dict(image_id=100+i,oam_id=oid,validation_fold=fold,
                crs='EPSG:3395', bounds=[1000000+i,1000000,1000100+i,1000100],
                meta=f'https://metadata.example/{oid}')))
    rows,groups,_=gate.qa.load_catalog(paths)
    prior=[dict(image_id=rid,oam_id=row['oam_id'],split=row['split'],alias=gate.qa.identity(row),
                envelope=groups[row['oam_id']]['wgs84_bounds']) for rid,row in rows.items() if rid>=100]
    def write(name,data):
        path=base/name
        path.write_bytes(producer.canonical(data))
        return dict(path=name,sha256=gate.qa.sha(path))
    freeze=write('freeze.json',dict(identity=dict(crop_plan_sha256=gate.CROP_PLAN_SHA)))
    cat={k:gate.qa.sha(path) for k,path in paths.items()}
    protected=write('protected.json',dict(scope='v5 and nested historical Val18 only', sources=prior,
        provenance=dict(freeze_sha256=freeze['sha256'],crop_plan_sha256=gate.CROP_PLAN_SHA,
                        metadata_sha256={f'{k}_meta.json':cat[k] for k in ('train','test')})))
    anchors=dict(FREEZE_SHA=freeze['sha256'], SUBINDEX_SHA=protected['sha256'])
    for name,pin in anchors.items():
        monkeypatch.setattr(gate,name,pin)
    a1=json.loads((base/'a1.json').read_bytes())
    a1['catalog_sha256']=cat
    contract=write('a1.json',a1)
    report=gate.qa.assess(root,evidence,base/'a1.json',contract['sha256'])
    report_ref=write('report.json',report)
    plan=dict(schema=gate.PLAN, source_generation=gate.qa.GENERATION, gt54=gate.GT54,
        storage_policy=gate.STORAGE_POLICY, windows_per_source={'train':1,'val':1},
        limits=dict(gate.MAXIMUMS,**gate.MINIMUMS),
        locations=dict(cache=str(root),evidence=str(evidence),output=str(out)),
        protected_roots=[str(base)],
        a1={role:dict(contract=contract,report=report_ref.copy()) for role in ('train','val')},
        protected=protected,historical_freeze=freeze,catalog_sha256=cat,
        allow_authenticity=sorted(gate.AUTH),
        sources=[dict(x,historical_train_reuse=False) for x in report['results']])
    return plan,write,(base,root,evidence,out),anchors


@pytest.mark.parametrize('problem',['anchor','crop','geometry','cohort','allowlist','hold','current_bytes','rights'])
def test_real_evidence_stops(fake_real, problem):
    plan,write,(base,root,evidence,_),_=fake_real
    if problem=='anchor':
        protected=json.loads((base/'protected.json').read_bytes())
        protected['sources'][0]['image_id']=777
        plan['protected']=write('protected.json',protected)
    elif problem=='crop':
        plan['sources'][0]['crop_xywh']=[0,0,224,224]
    elif problem=='geometry':
        plan['sources'][0]['crop_transform'][2] += 1
    elif problem=='current_bytes':
        row=plan['sources'][0]
        producer.source_paths(root,row)['annotation'].write_bytes(b'changed')
    elif problem=='rights':
        (evidence/(plan['sources'][0]['oam_id']+'-stac.json')).write_text('{}')
    else:
        report=json.loads((base/'report.json').read_bytes())
        if problem=='cohort':
            report['results'].pop()
        elif problem=='allowlist':
            report['eligible_ids']['train']=[]
        else:
            report['results'][0]['window_decision']['status']=gate.qa.HOLD
        plan['a1']['train']['report']=write('report.json',report)
    with pytest.raises(ValueError, match={
            'anchor':'HISTORICAL_ANCHOR','crop':'REAL_A1_SOURCE_BINDING','geometry':'REAL_A1_SOURCE_BINDING',
            'cohort':'REAL_A1_COHORT','allowlist':'REAL_A1_ALLOWLIST','hold':'REAL_A1_RESULT_BINDING',
            'current_bytes':'INPUT_PIN_MISMATCH','rights':'STAC_HASH_MISMATCH'}[problem]):
        gate.verify_evidence(plan,'0'*64,base,root,evidence)


@pytest.mark.skipif(sys.platform!='win32',reason='Windows Job and AllocationSize integration')
@pytest.mark.parametrize('synthetic',[2048],indirect=True)
def test_full_size_approved_fake_real_job_roundtrip(fake_real, monkeypatch):
    from d6a3_data import D6Data
    from d6a3_guard import subprocess
    from r2_train import make_plan

    plan,write,(base,root,evidence,out),anchors=fake_real
    start = subprocess.Popen
    pause = False
    processes=[]

    def fixture_job(args, **kwargs):
        # The real CLI cannot substitute the reviewed historical digests. Inject
        # fixture-only values inside the test child after its real Job handshake.
        command=args.index('-c')+1
        args[command]=args[command].replace(
            'from d6a32_gate import run_guarded; run_guarded',
            f'import d6a32_gate; d6a32_gate.__dict__.update({anchors!r}); d6a32_gate.run_guarded')
        if pause:
            args[command] = "import sys,time; sys.stdin.buffer.read(1); time.sleep(30)"
        process=start(args,**kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess,'Popen',fixture_job)
    def launch(resume=False):
        ref=write('plan.json',plan)
        token=write('approval.json',dict(schema=gate.APPROVAL,approved=True,plan_sha256=ref['sha256'],
                                         purpose='ONE_BOUNDED_REAL_OUTPUT_PILOT'))
        return gate.execute(base/ref['path'],ref['sha256'],base/token['path'],token['sha256'],
                            root,evidence,out,resume=resume)

    def hashes(directory):
        return {p.relative_to(directory):gate.qa.sha(p) for p in directory.rglob('*') if p.is_file()}

    pause=True
    plan['limits']['attempt_seconds']=plan['limits']['seconds']=1
    with pytest.raises(RuntimeError,match='deadline'):
        launch()
    assert processes[-1].poll() is not None and not out.exists()
    pause=False
    plan['limits']['attempt_seconds']=600
    plan['limits']['seconds']=120

    revised=copy.deepcopy(plan['locations'])
    plan['locations']['output']=str(base/'protected-output')
    with pytest.raises(ValueError,match='REAL_LOCATION_BINDING'):
        launch()
    plan['locations']=revised
    original=hashes(root),hashes(evidence)
    result=launch()
    assert result['peak_observed_allocation_bytes']>0
    assert result['attempt_measurements']['peak_rss_bytes']>0
    data=D6Data(out,result['manifest_sha256'])
    assert len(data.samples)==4
    schedule=make_plan(data,steps=4)
    for role in ('train','val'):
        assert data.patch(schedule[role][0],role)[0].shape==(1,3,224,224)
    frozen=hashes(out)
    assert launch(resume=True)['skipped_sources']==4
    assert hashes(out)==frozen and (hashes(root),hashes(evidence))==original
    print('Synthetic four-source 2048x2048 controls:',result,
          'logical bytes:',producer.tree_bytes(out))
    (next((out/'sources').iterdir())/'rgb.tif').write_bytes(b'changed')
    with pytest.raises(ValueError,match='PRODUCT_INTEGRITY_FAIL'):
        D6Data(out,result['manifest_sha256'])


def test_allocated_size_and_free_space_limits(tmp_path, monkeypatch):
    import d6a3_guard

    (tmp_path/'file').write_bytes(b'one')
    monkeypatch.setattr(d6a3_guard,'allocated_bytes',lambda _:4096)
    limits=dict(free_bytes=0,physical_bytes=4095,output_bytes=10)
    with pytest.raises(ValueError,match='allocation'):
        producer.storage_check(tmp_path,limits)
    limits['free_bytes']=2**60
    with pytest.raises(ValueError,match='free-space'):
        producer.storage_check(tmp_path,limits)


def test_real_detected_allocation_stop_retains_unverified_stage(fake_real, monkeypatch):
    import d6a3_guard

    plan,write,(base,root,evidence,out),_=fake_real
    ref=write('plan.json',plan)
    inputs=gate.verify_evidence(plan,ref['sha256'],base,root,evidence)
    monkeypatch.setattr(d6a3_guard,'allocated_bytes',
                        lambda path: plan['limits']['physical_bytes']+1 if path.suffix=='.tif' else 0)
    with pytest.raises(ValueError,match='observed file allocation'):
        producer.publish(base/ref['path'],ref['sha256'],root,out,inputs,real=True)
    assert not (out/'manifest.json').exists()
    assert len(list((out/'sources').glob('.stage-*'))) == 1
    before={p:gate.qa.sha(p) for p in out.rglob('*') if p.is_file()}
    monkeypatch.setattr(d6a3_guard,'allocated_bytes',lambda _:0)
    with pytest.raises(ValueError,match='UNVERIFIED_STAGE'):
        producer.publish(base/ref['path'],ref['sha256'],root,out,inputs,real=True,resume=True)
    assert {p:gate.qa.sha(p) for p in out.rglob('*') if p.is_file()} == before
