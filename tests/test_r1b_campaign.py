"""Offline safety tests: fake bounded readers, synthetic bytes; no network/images."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import r1b_campaign as m


@pytest.fixture
def setup(tmp_path):
    cache=tmp_path/'cache';(cache/'archives').mkdir(parents=True)
    metadata={}
    for name in ('train_meta.json','test_meta.json'):
        p=cache/'archives'/name;p.write_bytes(b'{}\n');metadata[name]=m.sha(p)
    pool=[dict(image_id=i,oam_id=f'source-{i}',split='val' if i%5==4 else 'train',official_split='train',validation_fold=0 if i%5==4 else 1) for i in range(30)]
    plan=dict(schema='ouem-r1b1-plan-v1',source_generation=m.GENERATION,
              plan=dict(initial_batch_ids=list(range(25)),pool=pool,completeness=dict(status='PASS')),
              inputs=dict(metadata_sha256=metadata))
    plan['plan_id']='ouem-r1b1-'+m.hash_bytes(m.encoded(plan))
    p=tmp_path/'plan.json';p.write_bytes(m.encoded(plan))
    return dict(root=tmp_path/'campaign',registry=tmp_path/'registry',plan_path=p,plan_sha=m.sha(p),cache=cache,campaign_id='offline-test')


def create(s,limits=None):m.Campaign.create(**s,limits=limits)
def open_campaign(s):return m.Campaign.open(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
def inspect(s):return m.Campaign.inspect(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
def reconcile(s,lock=None):return m.Campaign.reconcile(s['root'],s['registry'],s['plan_path'],s['plan_sha'],acknowledge_stopped_owner=True,expected_lock_sha=lock)


def test_reserve_receive_write_commit_resume_and_counters(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        c.reserve('http1',0,'train','stac','artifacts/0/stac.json',30,network_bytes=32)
        assert c.usage()['network_charged_bytes']==32
        body=c.receive('http1',io.BytesIO(b'{"ok":1}').read)
        c.write_bytes('http1',body);c.commit('http1')
        assert c.usage()['network_charged_bytes']==len(body)
        assert c.usage()['requests']==1
        with pytest.raises(m.Stop,match='settled'):c.commit('http1')
        with pytest.raises(m.Stop,match='duplicate'):c.reserve('http1',0,'train','report','artifacts/other.json',20)
    with open_campaign(s) as c:
        c.reserve('preview',4,'val','preview','artifacts/4/preview.png',20)
        c.write_bytes('preview',b'synthetic');c.commit('preview')
    report=inspect(s)
    assert report['status']=='CLEAN_PAUSED'
    assert report['operations']['preview']['file']['sha256']==m.hash_bytes(b'synthetic')
    assert report['usage']['disk_bytes']>len(body)+len(b'synthetic')
    assert report['usage']['unresolved_storage_reservation_bytes']==0
    assert len(report['contract']['source_files'])==25
    assert report['contract']['source_files']['0']['byte_validation']=='NOT_VERIFIED_R1B2B_REQUIRED'


@pytest.mark.parametrize('problem',['sha','id','metadata','split','ids','pool'])
def test_input_rejection(setup,problem):
    s=setup
    if problem=='sha':s['plan_sha']='f'*64
    elif problem=='id':
        with pytest.raises(m.Stop):m.inputs(s['plan_path'],s['plan_sha'],s['cache'],'wrong')
        return
    elif problem=='metadata':(s['cache']/'archives/train_meta.json').write_bytes(b'changed')
    else:
        d=json.loads(s['plan_path'].read_bytes());d.pop('plan_id')
        if problem=='split':d['plan']['pool'][0]['split']='val'
        elif problem=='ids':d['plan']['initial_batch_ids'][-1]=99
        else:d['plan']['initial_batch_ids']=list(range(30))
        d['plan_id']='ouem-r1b1-'+m.hash_bytes(m.encoded(d));s['plan_path'].write_bytes(m.encoded(d));s['plan_sha']=m.sha(s['plan_path'])
    with pytest.raises((m.Stop,KeyError)):create(s)


def test_contract_pin_and_permanent_registry_prevents_reset(setup):
    s=setup;create(s)
    replacement=dict(s,root=s['root'].with_name('other'),campaign_id='renamed')
    with pytest.raises(FileExistsError):create(replacement)
    with pytest.raises(m.Stop):m.Campaign.open(s['root'],s['registry'],s['plan_path'],'0'*64)
    contract=s['root']/'contract.json';data=json.loads(contract.read_bytes());data['limits']['network_bytes']+=1;contract.write_bytes(m.encoded(data))
    with pytest.raises(m.Stop,match='SHA'):inspect(s)


def test_preoperation_network_response_request_storage_and_free_space_limits(setup,monkeypatch):
    s=setup;limits=dict(m.DEFAULT_LIMITS,network_bytes=20,requests=1,response_bytes=16,storage_bytes=1024**2);create(s,limits)
    with open_campaign(s) as c:
        for kwargs in [dict(network_bytes=17),dict(network_bytes=21)]:
            with pytest.raises(m.Stop):c.reserve('bad',0,'train','stac','artifacts/bad',20,**kwargs)
        with pytest.raises(m.Stop):c.reserve('big',0,'train','report','artifacts/big',1024**2)
        with pytest.raises(m.Stop):c.reserve('rgb',0,'train','geotiff','artifacts/rgb',20,network_bytes=1)
        c.reserve('one',0,'train','stac','artifacts/one',20,network_bytes=16)
        c.write_bytes('one',c.receive('one',io.BytesIO(b'abc').read));c.commit('one')
        with pytest.raises(m.Stop):c.reserve('two',0,'train','stac','artifacts/two',20,network_bytes=1)
        assert len(c.state['operations'])==1
        class Space:free=0
        monkeypatch.setattr(m.shutil,'disk_usage',lambda p:Space())
        with pytest.raises(m.Stop,match='space'):c.reserve('disk',0,'train','report','artifacts/disk',20)


def test_network_cumulative_failures_partial_bytes_full_charge_and_no_auto_retry(setup):
    s=setup;create(s)
    with pytest.raises(RuntimeError), open_campaign(s) as c:
        c.reserve('partial',0,'train','stac','artifacts/partial.json',20,network_bytes=16)
        calls=[]
        def reader(n):
            calls.append(n)
            if len(calls)==1:return b'abc'
            raise OSError('interrupted after possible hidden receive')
        with pytest.raises(OSError):c.receive('partial',reader)
        assert c.usage()['network_charged_bytes']==16 and c.usage()['requests']==1
        assert c.state['operations']['partial']['received_bytes']==3
        with pytest.raises(m.Stop):c.receive('partial',reader)
        with pytest.raises(m.Stop):c.reserve('retry',0,'train','stac','artifacts/retry',20,network_bytes=16)
        # Exception exit releases only our lock, never the unresolved reservation.
        raise RuntimeError('stop')
    assert inspect(s)['status']=='STOP'
    with pytest.raises(m.Stop):open_campaign(s)
    reconcile(s)
    with open_campaign(s) as c:
        assert c.usage()['network_charged_bytes']==16 and c.usage()['requests']==1
        with pytest.raises(m.Stop):c.reserve('partial',0,'train','report','artifacts/new',20)


def test_partial_write_failure_preserved_and_reconciled(setup,monkeypatch):
    s=setup;create(s);original=m.durable
    with pytest.raises(OSError):
        with open_campaign(s) as c:
            c.reserve('write',0,'train','preview','artifacts/out.png',20)
            def fail(path,data):
                if path.suffix=='.partial':path.write_bytes(data[:3]);raise OSError('disk interruption')
                return original(path,data)
            monkeypatch.setattr(m,'durable',fail)
            c.write_bytes('write',b'abcdef')
    report=inspect(s);assert report['partial_files'][0]['bytes']==3
    assert report['usage']['unresolved_storage_reservation_bytes']==37
    monkeypatch.setattr(m,'durable',original);reconcile(s)
    report=inspect(s)
    assert report['usage']['unresolved_storage_reservation_bytes']==0
    assert report['operations']['write']['retained_files'][0]['sha256']==m.hash_bytes(b'abc')


def test_crash_lock_subprocess_and_explicit_reconciliation(setup):
    s=setup;create(s)
    code="""import sys,os;sys.path.insert(0,sys.argv[1]);import r1b_campaign as m
c=m.Campaign.open(*sys.argv[2:]);c.reserve('crash',0,'train','stac','artifacts/crash.json',20,network_bytes=16);os._exit(7)
"""
    result=subprocess.run([sys.executable,'-c',code,str(Path(m.__file__).parent),str(s['root']),str(s['registry']),str(s['plan_path']),s['plan_sha']],capture_output=True)
    assert result.returncode==7,result.stderr.decode()
    report=inspect(s);lock=report['lock_sha256'];assert lock and report['status']=='STOP'
    with pytest.raises(m.Stop):open_campaign(s)
    with pytest.raises(m.Stop):m.Campaign.reconcile(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
    with pytest.raises(m.Stop):reconcile(s,'f'*64)
    assert (s['root']/'campaign.lock').exists()
    reconcile(s,lock)
    with open_campaign(s) as c:assert c.usage()['network_charged_bytes']==16
    assert list((s['root']/'recovered-locks').glob('*.json'))


@pytest.mark.parametrize('problem',['journal','head','artifact','missing','unknown'])
def test_corruption_sha_and_unowned_files_fail_closed(setup,problem):
    s=setup;create(s)
    with open_campaign(s) as c:
        c.reserve('ok',0,'train','report','artifacts/ok.json',20);c.write_bytes('ok',b'good');c.commit('ok')
        entry=c.entry
    if problem=='journal':(s['root']/'events/00000001.json').write_bytes(b'broken')
    elif problem=='head':(entry/'head.json').write_bytes(b'{"sequence":999,"sha256":"bad"}')
    elif problem=='artifact':(s['root']/'artifacts/ok.json').write_bytes(b'evil')
    elif problem=='missing':(s['root']/'artifacts/ok.json').unlink()
    else:(s['root']/'artifacts/unknown').write_bytes(b'unknown')
    with pytest.raises((ValueError,OSError)):open_campaign(s)


def test_locks_targets_paths_and_overwrite(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        with pytest.raises(FileExistsError):open_campaign(s)
        for image,split in [(25,'train'),(0,'val')]:
            with pytest.raises(m.Stop):c.reserve('bad',image,split,'report','artifacts/bad',20)
        for path in ('../outside','artifacts/../outside','C:/outside','artifacts/CON','artifacts/out:stream'):
            with pytest.raises(m.Stop):c.reserve('bad',0,'train','report',path,20)
        c.reserve('ok',0,'train','report','artifacts/ok',20);c.write_bytes('ok',b'abc');c.commit('ok')
        with pytest.raises(m.Stop):c.reserve('overwrite',0,'train','report','artifacts/ok',20)
    protected=dict(s,root=s['cache']/'bad',registry=s['registry'].with_name('reg2'))
    with pytest.raises(m.Stop):create(protected)


def test_cap_eof_and_reader_overrun_never_claim_success(setup):
    s=setup;create(s)
    with pytest.raises(m.Stop):
        with open_campaign(s) as c:
            c.reserve('full',0,'train','stac','artifacts/full',8,network_bytes=8)
            c.receive('full',io.BytesIO(b'12345678').read)
    assert inspect(s)['status']=='STOP'


def test_cumulative_network_reservations_reject_before_reader(setup):
    s=setup;create(s,dict(m.DEFAULT_LIMITS,network_bytes=20,response_bytes=16))
    with open_campaign(s) as c:
        c.reserve('one',0,'train','stac','artifacts/one',20,network_bytes=16)
        c.write_bytes('one',c.receive('one',io.BytesIO(b'abc').read));c.commit('one')
        c.reserve('two',0,'train','stac','artifacts/two',20,network_bytes=15)
        assert c.usage()['network_charged_bytes']==18
        with pytest.raises(m.Stop,match='Network budget'):c.reserve('three',0,'train','stac','artifacts/three',20,network_bytes=4)
        assert 'three' not in c.state['operations']
        c.write_bytes('two',c.receive('two',io.BytesIO(b'abc').read));c.commit('two')
        assert c.usage()['network_charged_bytes']==6


def test_uncooperative_reader_overrun_is_logged_and_fatal(setup):
    s=setup;create(s)
    with pytest.raises(m.Stop):
        with open_campaign(s) as c:
            c.reserve('bad',0,'train','stac','artifacts/bad',20,network_bytes=8)
            c.receive('bad',lambda n:b'x'*12)
    report=inspect(s)
    assert report['usage']['network_charged_bytes']==12
    with pytest.raises(m.Stop,match='Fatal'):reconcile(s)


def test_head_publication_interruption_keeps_old_records_and_stops(setup,monkeypatch):
    s=setup;create(s);first=(s['root']/'events/00000001.json').read_bytes()
    with pytest.raises(OSError):
        with open_campaign(s) as c:
            def fail(a,b):raise OSError('atomic head publication interrupted')
            monkeypatch.setattr(m.os,'replace',fail)
            c.reserve('orphan',0,'train','report','artifacts/orphan',20)
    assert (s['root']/'events/00000001.json').read_bytes()==first
    assert (s['root']/'campaign.lock').exists()
    with pytest.raises(m.Stop):inspect(s)


def test_cli_initialization_inspection_resume_and_no_overwrite(setup,capsys):
    s=setup
    args=['--root',str(s['root']),'--registry',str(s['registry']),'--plan',str(s['plan_path']),'--plan-sha256',s['plan_sha']]
    assert m.main(['init',*args,'--cache',str(s['cache']),'--campaign-id','cli'])==0
    assert json.loads(capsys.readouterr().out)['contract']['execution_gate']=='OFFLINE_SAFETY_ONLY_NOT_REAL_DATA_APPROVAL'
    assert m.main(['resume-check',*args])==0;capsys.readouterr()
    assert m.main(['inspect',*args])==0;capsys.readouterr()
    events=list((s['root']/'events').iterdir())
    with pytest.raises(m.Stop):m.main(['reconcile',*args,'--plan-id','wrong','--acknowledge-stopped-owner'])
    assert list((s['root']/'events').iterdir())==events
    with pytest.raises(m.Stop):m.main(['init',*args,'--cache',str(s['cache']),'--campaign-id','again'])


def test_reparse_path_guard_without_symlink_privilege(tmp_path,monkeypatch):
    from types import SimpleNamespace
    target=tmp_path/'reparse';target.write_bytes(b'x');original=Path.lstat
    def fake(path,*args,**kwargs):
        if path==target:return SimpleNamespace(st_mode=m.stat.S_IFREG,st_file_attributes=1024)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',fake)
    with pytest.raises(m.Stop,match='reparse'):m.ordinary(target)


def test_journal_records_operation_deltas_not_growing_snapshots(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        for i in range(5):
            key=f'file{i}'
            c.reserve(key,0,'train','report',f'artifacts/{key}.json',20)
            c.write_bytes(key,b'{}');c.commit(key)
    records=[json.loads(p.read_bytes()) for p in sorted((s['root']/'events').iterdir())]
    assert all(len(e['operations'])<=1 for e in records)
    assert len(inspect(s)['operations'])==5


def test_additional_legacy_output_protection(setup,tmp_path):
    s=dict(setup);old=tmp_path/'old-checkpoints';old.mkdir();(old/'keep.pth').write_bytes(b'keep')
    s['root']=old/'new-campaign'
    with pytest.raises(m.Stop,match='Protected'):m.Campaign.create(**s,protected_roots=[old])
    assert (old/'keep.pth').read_bytes()==b'keep' and not s['root'].exists()


def test_in_memory_budget_or_counter_changes_are_rejected(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        original=copy.deepcopy(c.contract);c.contract['limits']['storage_bytes']+=1000
        with pytest.raises(m.Stop,match='contract changed'):c.reserve('bad',0,'train','report','artifacts/bad',20)
        c.contract=original
        c.reserve('ok',0,'train','report','artifacts/ok',20);c.write_bytes('ok',b'{}');c.commit('ok')
        c.state['operations']['ok']['received_bytes']=100
        with pytest.raises(m.Stop,match='Unjournaled'):c.reserve('bad2',0,'train','report','artifacts/bad2',20)
        c.state=copy.deepcopy(c._persisted_state)


def test_changed_owner_lock_blocks_operations_and_is_not_deleted(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        lock=s['root']/'campaign.lock';owned=lock.read_bytes();lock.write_bytes(b'other owner')
        with pytest.raises(m.Stop,match='Owner lock'):c.reserve('bad',0,'train','report','artifacts/bad',20)
        with pytest.raises(m.Stop,match='Owner lock'):c.__exit__(None,None,None)
        assert lock.read_bytes()==b'other owner'
        lock.write_bytes(owned)  # test fixture restores its own lock; never automatic recovery


def test_written_receipt_is_durable_and_normal_commit_matches_it(setup):
    s=setup;create(s)
    with open_campaign(s) as c:
        c.reserve('receipt',0,'train','report','artifacts/receipt',20)
        c.write_bytes('receipt',b'good')
        receipt=dict(bytes=4,sha256=m.hash_bytes(b'good'))
        assert inspect(s)['operations']['receipt']['written_file']==receipt
        c.commit('receipt')
        assert c.state['operations']['receipt']['file']==receipt
    assert inspect(s)['status']=='CLEAN_PAUSED'


def test_reserved_commit_rejected_even_if_a_partial_was_supplied_outside_api(setup):
    s=setup;create(s)
    with pytest.raises(m.Stop):
        with open_campaign(s) as c:
            c.reserve('unwritten',0,'train','report','artifacts/unwritten',20)
            (s['root']/'artifacts/unwritten.partial').write_bytes(b'not written by API')
            with pytest.raises(m.Stop,match='WRITTEN'):c.commit('unwritten')
    report=inspect(s)
    assert report['operations']['unwritten']['status']=='RESERVED'
    assert not (s['root']/'artifacts/unwritten').exists()


@pytest.mark.parametrize('problem',['same_size_change','size_change','missing'])
def test_changed_partial_cannot_be_published(setup,problem):
    s=setup;create(s)
    with pytest.raises(m.Stop):
        with open_campaign(s) as c:
            c.reserve('edited',0,'train','report','artifacts/edited',20);c.write_bytes('edited',b'good')
            partial=s['root']/'artifacts/edited.partial'
            if problem=='missing':partial.unlink()
            else:partial.write_bytes(b'evil' if problem=='same_size_change' else b'changed length')
            c.commit('edited')
    report=inspect(s)
    assert report['status']=='STOP' and report['operations']['edited']['status']=='INTERRUPTED'
    assert report['operations']['edited']['written_file']['sha256']==m.hash_bytes(b'good')
    assert not (s['root']/'artifacts/edited').exists()


def test_interrupted_commit_rejected_and_stale_reconcile_guard_is_visible(setup):
    s=setup;create(s)
    with pytest.raises(RuntimeError), open_campaign(s) as c:
        c.reserve('broken',0,'train','stac','artifacts/broken',20,network_bytes=16)
        with pytest.raises(OSError):c.receive('broken',lambda n:(_ for _ in ()).throw(OSError('interrupt')))
        with pytest.raises(m.Stop):c.commit('broken')
        raise RuntimeError('stop interrupted owner')
    with pytest.raises(m.Stop):
        with open_campaign(s) as c:c.commit('broken')
    reconcile(s)
    report=inspect(s);assert report['status']=='CLEAN_PAUSED'
    entry=s['registry']/report['contract']['registry_binding'];guard=entry/'reconcile.lock'
    guard.write_bytes(b'crashed after campaign lock release, before guard cleanup')
    before=guard.read_bytes();report=inspect(s)
    assert report['status']=='STOP'
    assert report['stop_reasons']==['RECONCILIATION_LOCK_PRESENT']
    assert report['reconcile_lock_sha256']==m.hash_bytes(before)
    assert guard.read_bytes()==before
    with pytest.raises(m.Stop,match='reconciliation'):open_campaign(s)
