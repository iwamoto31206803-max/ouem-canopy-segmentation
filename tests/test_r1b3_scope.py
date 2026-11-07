"""Synthetic 150-record sources only; real Campaigns and sockets are never used."""
import copy
import json
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import r1b_campaign as c
import r1b_qa_runner as q
import r1b3_scope as m
from r1b_candidate_plan import identity
from test_r1b_qa_runner import approval,Factory,Response,stac_body


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*a,**k):raise AssertionError('REAL NETWORK FORBIDDEN')
    monkeypatch.setattr(socket,'create_connection',fail)
    monkeypatch.setattr(socket.socket,'connect',fail)


def snapshot(root):return {str(p):c.sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def setup(tmp_path):
    cache=tmp_path/'cache';archives=cache/'archives';archives.mkdir(parents=True)
    (cache/'SYNTHETIC_QA_FIXTURE.txt').write_bytes(q.FIXTURE_MARKER)
    source=cache/'extracted/dataset/holdout';(source/'images').mkdir(parents=True);(source/'masks').mkdir()
    rows=[];pool=[]
    for i in range(150):
        t=from_origin(i*100000,10000,.1,.1);bounds=[i*100000,9999.2,i*100000+.8,10000]
        row=dict(image_id=i,oam_id=f'source-{i}',validation_fold=0 if i%5==4 else 1,
            license='CC-BY 4.0',width=8,height=8,crs='EPSG:3395',bounds=bounds,
            meta=f'https://example.org/source/{i}_meta.json')
        rows.append(row)
        pool.append(dict(image_id=i,oam_id=row['oam_id'],official_split='train',validation_fold=row['validation_fold'],
            split='val' if i%5==4 else 'train',row_license=row['license'],shape=[8,8],
            original_source_identity_sha256=identity(row),representative_metadata_bounds_wgs84=list(transform_bounds('EPSG:3395',4326,*bounds))))
        with rasterio.open(source/'images'/f'source-{i}_{i}.tif','w',driver='GTiff',width=8,height=8,count=3,dtype='uint8',crs='EPSG:3395',transform=t) as dst:
            dst.write(np.full((3,8,8),100,np.uint8))
        ann=np.zeros((8,8,3),np.uint8);ann[:,4:,0]=1
        Image.fromarray(ann).save(source/'masks'/f'source-{i}_{i}.png')
    (archives/'train_meta.json').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (archives/'test_meta.json').write_text('',encoding='utf-8')
    coco=archives/'dataset/holdout/train.json';coco.parent.mkdir(parents=True)
    coco.write_text(json.dumps(dict(annotations=[dict(image_id=i,id=1000+i,category_id=1,iscrowd=0) for i in range(150)])))
    plan=dict(schema='ouem-r1b1-plan-v1',source_generation=c.GENERATION,
        plan=dict(pool=pool,initial_batch_ids=list(range(25)),completeness=dict(status='PASS')),
        inputs=dict(metadata_sha256={n:c.sha(archives/n) for n in ('train_meta.json','test_meta.json')}))
    plan['plan_id']='ouem-r1b1-'+c.hash_bytes(c.encoded(plan));parent=tmp_path/'parent.json';parent.write_bytes(c.encoded(plan))
    legacy=tmp_path/'legacy';registry=tmp_path/'legacy-registry'
    c.Campaign.create(legacy,registry,parent,c.sha(parent),cache,'old-synthetic')
    with c.Campaign.open(legacy,registry,parent,c.sha(parent)) as campaign:
        for i in range(25):
            split='val' if i%5==4 else 'train'
            hashes={k:c.sha(source/folder/f'source-{i}_{i}.{ext}') for k,folder,ext in
                [('image_sha256','images','tif'),('annotation_sha256','masks','png')]}
            report=dict(image_id=i,split=split,source_generation=c.GENERATION,technical_status='PASS',r1_acceptance='HOLD',checks=dict(source_hashes=hashes))
            for kind,data in [('report',c.encoded(report)),('preview',b'synthetic-preview')]:
                op=f'qa-{kind}-{i}';path=f'artifacts/{op}.json'
                campaign.reserve(op,i,split,kind,path,len(data));campaign.write_bytes(op,data);campaign.commit(op)
    scope=m.build_scope(parent,c.sha(parent),cache,legacy,registry,plan['plan_id'])
    path=tmp_path/'scope.json';path.write_bytes(c.encoded(scope))
    return dict(cache=cache,parent=parent,parent_sha=c.sha(parent),plan=plan,legacy=legacy,legacy_registry=registry,
        scope=scope,scope_path=path,root=tmp_path/'new-campaign',registry=tmp_path/'new-registry')


def create(s,limits=None):m.ScopeCampaign.create(s['root'],s['registry'],s['scope_path'],c.sha(s['scope_path']),s['cache'],'new-synthetic',limits=limits)
def opened(s):return m.ScopeCampaign.open(s['root'],s['registry'],s['scope_path'],c.sha(s['scope_path']))
def inspect(s):return m.ScopeCampaign.inspect(s['root'],s['registry'],s['scope_path'],c.sha(s['scope_path']))
def save_scope(s,value):
    value.pop('scope_id',None);value['scope_id']='ouem-r1b3-'+c.hash_bytes(c.encoded(value));s['scope_path'].write_bytes(c.encoded(value))


def test_real_parent_scope_derivation_reads_only_repository_plan():
    p=Path(__file__).parents[1]/'docs/implementation/R1B1_CANDIDATE_PLAN.json';raw=p.read_bytes();plan=json.loads(raw)
    selected=m.remaining(plan)
    assert plan['plan_id']=='ouem-r1b1-fbb4ed018dc3cfded676d6ba09229ef7ccfb80fa7bef9b9b67a1df84864fcc5a'
    assert len(selected)==125 and sum(r['split']=='train' for r in selected)==100
    assert [r['image_id'] for r in selected]==[r['image_id'] for r in plan['plan']['pool'][25:]]
    assert not set(plan['plan']['initial_batch_ids'])&{r['image_id'] for r in selected}
    assert p.read_bytes()==raw


def test_scope_determinism_and_metadata_only_reads(setup,monkeypatch):
    s=setup;before=snapshot(s['legacy']);original=snapshot(s['cache'])
    def fail(*a,**k):raise AssertionError('PIXEL DECODE FORBIDDEN')
    monkeypatch.setattr(q.Image,'open',fail);monkeypatch.setattr(q.rasterio,'open',fail)
    repeated=m.build_scope(s['parent'],s['parent_sha'],s['cache'],s['legacy'],s['legacy_registry'])
    assert c.encoded(repeated)==c.encoded(s['scope'])
    proxy,targets,_=m.scope_inputs(s['scope_path'],c.sha(s['scope_path']),s['cache'])
    assert list(targets)==[str(i) for i in range(25,150)]
    assert proxy['plan']['initial_batch_ids']==list(range(25,150))
    assert len(s['scope']['reused_evidence']['samples'])==25
    output=s['scope_path'].with_name('cli-scope.json')
    args=['build-scope','--parent-plan',str(s['parent']),'--parent-sha256',s['parent_sha'],
        '--parent-plan-id',s['plan']['plan_id'],'--cache',str(s['cache']),'--legacy-root',str(s['legacy']),
        '--legacy-registry',str(s['legacy_registry']),'--output',str(output)]
    assert m.main(args)==0 and output.read_bytes()==c.encoded(s['scope'])
    assert not s['root'].exists() and not s['registry'].exists()
    with pytest.raises(FileExistsError):m.main(args)
    assert snapshot(s['legacy'])==before and snapshot(s['cache'])==original


def test_scope_tampering_and_missing_source_rejected(setup):
    s=setup
    with pytest.raises(c.Stop,match='Scope file SHA'):m.scope_inputs(s['scope_path'],'0'*64,s['cache'])
    def change(v,case):
        if case=='missing':v['samples'].pop()
        elif case=='duplicate':v['samples'][1]=copy.deepcopy(v['samples'][0])
        elif case=='order':v['samples'][0],v['samples'][1]=v['samples'][1],v['samples'][0]
        elif case=='split':v['samples'][0]['split']='val'
        elif case=='source':v['samples'][0]['oam_id']='other'
        elif case=='filename':v['samples'][0]['source_files']['rgb']='wrong.tif'
        elif case=='reuse':v['reused_evidence']['samples'][0]['report']['sha256']='0'*64
        elif case=='parent':v['parent_plan']['sha256']='0'*64
        elif case=='parent-id':v['parent_plan']['plan_id']='wrong'
        elif case=='extra':v['samples'].append(copy.deepcopy(v['samples'][0]))
    for case in ['missing','duplicate','order','split','source','filename','reuse','parent','parent-id','extra']:
        v=copy.deepcopy(s['scope']);change(v,case);save_scope(s,v)
        with pytest.raises(c.Stop):m.scope_inputs(s['scope_path'],c.sha(s['scope_path']),s['cache'])
    save_scope(s,copy.deepcopy(s['scope']))
    source=s['cache']/'extracted/dataset/holdout/images/source-25_25.tif';source.rename(source.with_suffix('.held'))
    with pytest.raises(c.Stop,match='Missing scoped'):m.scope_inputs(s['scope_path'],c.sha(s['scope_path']),s['cache'])


def test_pool_roles_sources_and_legacy_evidence_fail_closed(setup):
    s=setup
    for field,value in [('official_split','test'),('validation_fold',-1),('row_license',None),('oam_id',s['plan']['plan']['pool'][0]['oam_id'])]:
        p=copy.deepcopy(s['plan']);p['plan']['pool'][25][field]=value
        with pytest.raises(c.Stop):m.remaining(p)
    path=s['cache']/'extracted/dataset/holdout/images/source-0_0.tif';path.write_bytes(b'changed synthetic original')
    with pytest.raises(c.Stop,match='Legacy source SHA'):m.scope_inputs(s['scope_path'],c.sha(s['scope_path']),s['cache'])


def test_metadata_holdout_source_alias_rejected_without_test_pixels(setup):
    s=setup
    row=json.loads((s['cache']/'archives/train_meta.json').read_text(encoding='utf-8').splitlines()[25])
    row.update(image_id=200,oam_id='holdout-alias',validation_fold=-1)
    (s['cache']/'archives/test_meta.json').write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(c.Stop,match='metadata mismatch'):m.selected_records(s['plan'],s['cache'])
    assert not (s['cache']/'extracted/dataset/holdout/images/holdout-alias_200.tif').exists()


def test_campaign_zero_network_limits_claim_and_v1_compatibility(setup):
    s=setup;old=snapshot(s['legacy']);create(s)
    v=inspect(s);assert v['status']=='CLEAN_PAUSED' and v['contract']['limits']==m.LIMITS
    assert len(v['contract']['targets'])==125 and len(v['contract']['reused_evidence']['samples'])==25
    with opened(s) as campaign:
        for kind in ['stac','geotiff']:
            with pytest.raises(c.Stop):campaign.reserve('denied',25,'train',kind,'artifacts/denied',10,network_bytes=1 if kind=='stac' else 0)
        with pytest.raises(c.Stop):campaign.reserve('old',0,'train','report','artifacts/old',10)
        assert not campaign.state['operations']
    with pytest.raises(FileExistsError):m.ScopeCampaign.create(s['root'].with_name('renamed'),s['registry'],s['scope_path'],c.sha(s['scope_path']),s['cache'],'renamed')
    with pytest.raises(c.Stop):c.Campaign.open(s['root'],s['registry'],s['scope_path'],c.sha(s['scope_path']))
    with c.Campaign.open(s['legacy'],s['legacy_registry'],s['parent'],s['parent_sha']) as legacy:
        assert legacy.contract['limits']==c.DEFAULT_LIMITS and len(legacy.contract['targets'])==25
    assert snapshot(s['legacy'])==old
    with pytest.raises(c.Stop):c.Campaign.checked_limits(m.LIMITS)
    for limits in [dict(m.LIMITS,network_bytes=1),dict(m.LIMITS,requests=1),dict(m.LIMITS,storage_bytes=256*1024**2+1)]:
        with pytest.raises(c.Stop):m.ScopeCampaign.checked_limits(limits)


def scoped_runner(s,campaign,ids):
    path,digest=approval(s,campaign,dict(allowed_image_ids=ids,scopes=['qa_local','qa_preview','qa_report'],
        scope_id=campaign.contract['scope_id'],scope_sha256=campaign.contract['scope_sha256']))
    return q.Runner(campaign,s['scope_path'],path,digest,test_mode=True,connection_factory=Factory(Response(stac_body())))


def test_five_batch_approvals_share_budget_and_refuse_duplicates(setup):
    s=setup;old=snapshot(s['legacy']);original=snapshot(s['cache']);create(s);disk=[]
    for i in range(5):
        with opened(s) as campaign:
            ids=list(range(25+i*25,50+i*25));r=scoped_runner(s,campaign,ids)
            assert r.approval.select_ids(r.order)==ids
            result=r.run_batch([ids[0]],run_id=f'batch-{i}')
            assert result['r1_acceptance']=='HOLD'
            report=json.loads((s['root']/f'artifacts/reports/qa-report-{ids[0]}.json').read_bytes())
            assert report['coverage_review']['no_data_classification']=='NOT_DETERMINED'
            disk.append(campaign.usage()['disk_bytes'])
    assert disk==sorted(set(disk)) and inspect(s)['usage']['requests']==0
    with opened(s) as campaign:
        r=scoped_runner(s,campaign,list(range(25,50)))
        with pytest.raises(c.Stop,match='duplicate QA'):r.run_batch([25],run_id='another')
        with pytest.raises(c.Stop,match='IMAGE_NOT_IN'):r.run_one(0,'train')
        for ids in [list(range(25,51)),[25,50]]:
            with pytest.raises(c.Stop,match='within one fixed batch'):scoped_runner(s,campaign,ids)
    assert snapshot(s['legacy'])==old and snapshot(s['cache'])==original


def test_approval_scope_binding_and_cli_guard(setup,monkeypatch):
    s=setup;create(s)
    with opened(s) as campaign:
        path,digest=approval(s,campaign,dict(allowed_image_ids=[25],scopes=['qa_local','qa_preview','qa_report'],scope_id='wrong',scope_sha256=campaign.contract['scope_sha256']))
        with pytest.raises(c.Stop,match='scope binding'):q.Approval(campaign,path,digest,test_mode=True)
        path,digest=approval(s,campaign,dict(allowed_image_ids=[25],scopes=['qa_local','stac_network'],scope_id=campaign.contract['scope_id'],scope_sha256=campaign.contract['scope_sha256']))
        with pytest.raises(c.Stop,match='local-only'):q.Approval(campaign,path,digest,test_mode=True)
        r=scoped_runner(s,campaign,[25]);path=r.approval.path;digest=r.approval.sha256
    original=q.Approval
    monkeypatch.setattr(q,'Approval',lambda campaign,path,digest:original(campaign,path,digest,test_mode=True))
    def fail(*a,**k):raise AssertionError('UNAPPROVED IMAGE ACCESS')
    monkeypatch.setattr(q,'Runner',fail)
    assert m.main(['check-approval','--root',str(s['root']),'--registry',str(s['registry']),'--plan',str(s['scope_path']),
        '--plan-sha256',c.sha(s['scope_path']),'--approval',str(path),'--approval-sha256',digest,'--ids','25'])==0
    with pytest.raises(c.Stop,match='IMAGE_NOT_IN'):
        m.main(['run','--execute','--root',str(s['root']),'--registry',str(s['registry']),'--plan',str(s['scope_path']),
            '--plan-sha256',c.sha(s['scope_path']),'--approval',str(path),'--approval-sha256',digest,'--ids','0'])


def test_storage_and_interrupted_partial_stop(setup):
    s=setup;create(s,dict(m.LIMITS,storage_bytes=1024**2))
    with pytest.raises(c.Stop,match='Pending operation'),opened(s) as campaign:
        with pytest.raises(c.Stop,match='budget'):campaign.reserve('too-large',25,'train','report','artifacts/large',1024**2)
        campaign.reserve('pending',25,'train','report','artifacts/pending.json',32)
        campaign.write_bytes('pending',b'partial awaiting commit')
    v=inspect(s);assert v['status']=='STOP' and v['usage']['unresolved_operations']==['pending'] and v['partial_files']
    with pytest.raises(c.Stop):opened(s)
    entry=s['registry']/v['contract']['registry_binding'];(entry/'reconcile.lock').write_bytes(b'synthetic crash')
    assert 'RECONCILIATION_LOCK_PRESENT' in inspect(s)['stop_reasons']


@pytest.mark.parametrize('value',[0,1])
def test_small_wedge_white_roof_flags_preserve_uniform_targets(value):
    rgb=np.full((100,100,3),100,np.uint8);rgb[0,:2]=0;rgb[1,0]=0;rgb[0:5,95:]=255
    mask=np.full((100,100),value,np.uint8);before=mask.copy();original=rgb.copy()
    black=q.component_report(np.all(rgb==0,axis=2),mask)[0]
    d=m.coverage_diagnostics(rgb,mask,black)
    assert black['border_connected_pixels']/mask.size<.01
    assert d['status']=='REVIEW_REQUIRED' and set(d['flags'])=={'EDGE_CONNECTED_BLACK','EDGE_CONNECTED_WHITE'}
    assert d['shape_candidates'] and d['no_data_classification']=='NOT_DETERMINED' and not d['valid_mask_generated']
    assert np.array_equal(mask,before) and np.array_equal(rgb,original)
