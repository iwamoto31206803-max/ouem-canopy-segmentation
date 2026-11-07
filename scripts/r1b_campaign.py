"""Offline R1b campaign safety API. No HTTP client, image QA or acquisition CLI."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from m1_7_dataset import encoded, sha, GENERATION

SCHEMA='ouem-r1b-campaign-v1'
DEFAULT_LIMITS=dict(network_bytes=2*1024**2,requests=25,response_bytes=65536,storage_bytes=94*1024**2)
CONTROL_MARGIN=65536
KINDS={'stac','preview','geotiff','report','temporary'}
TERMINAL={'SUCCESS','FAILED','RECONCILED_FULL_CHARGE'}


class Stop(ValueError):
    """Fail closed; inspect retained artifacts before any further operation."""


def hash_bytes(data):return hashlib.sha256(data).hexdigest()


def ordinary(path):
    for p in [Path(path),*Path(path).parents]:
        if p.exists() or p.is_symlink():
            s=p.lstat()
            if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&1024:
                raise Stop('Symlink/reparse point forbidden')


def relative(name):
    if not isinstance(name,str) or '\\' in name or ':' in name:raise Stop('Unsafe output path')
    for part in name.split('/'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',part) or part.endswith('.') or part.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
            raise Stop('Unsafe output path')
    if not name.startswith('artifacts/'):raise Stop('Output must be in campaign artifacts')
    return name


def durable(path,data):
    ordinary(path)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def inputs(plan_path,expected_sha,cache,expected_id=None):
    ordinary(plan_path);ordinary(cache)
    raw=Path(plan_path).read_bytes()
    if hash_bytes(raw)!=expected_sha:raise Stop('Plan file SHA mismatch')
    plan=json.loads(raw);core={k:v for k,v in plan.items() if k!='plan_id'}
    if plan.get('schema')!='ouem-r1b1-plan-v1' or plan.get('source_generation')!=GENERATION or plan.get('plan_id')!='ouem-r1b1-'+hash_bytes(encoded(core)):
        raise Stop('Plan ID/schema/generation mismatch')
    if expected_id is not None and plan['plan_id']!=expected_id:raise Stop('Plan ID mismatch')
    ids=plan['plan']['initial_batch_ids'];pool=plan['plan']['pool']
    if len(ids)!=25 or len(set(ids))!=25 or plan['plan']['completeness']['status']!='PASS':raise Stop('Complete fixed initial 25 required')
    candidates={r['image_id']:r for r in pool}
    targets={str(i):candidates[i]['split'] for i in ids}
    if list(ids)!=[r['image_id'] for r in pool[:25]] or len(candidates)!=len(pool) or any(type(i) is not int for i in ids):raise Stop('Initial cohort mismatch')
    for i in ids:
        r=candidates[i];fold=r['validation_fold']
        if not isinstance(r.get('oam_id'),str) or not re.fullmatch('[A-Za-z0-9-]+',r['oam_id']):raise Stop('Invalid source file identity')
        if r['official_split']!='train' or type(fold) is not int or fold not in range(5) or targets[str(i)]!=('val' if fold==0 else 'train'):
            raise Stop('Target split/fold mismatch')
    if sum(v=='train' for v in targets.values())!=20 or sum(v=='val' for v in targets.values())!=5:raise Stop('Initial split quota mismatch')
    metadata=plan['inputs']['metadata_sha256']
    if set(metadata)!={'train_meta.json','test_meta.json'}:raise Stop('Metadata pins required')
    for name,digest in metadata.items():
        path=Path(cache)/'archives'/name;ordinary(path)
        if sha(path)!=digest:raise Stop('Metadata SHA mismatch')
    return plan,targets,metadata


class Campaign:
    def __init__(self,root,registry):
        self.root=Path(root).absolute();self.registry=Path(registry).absolute();self.closed=True;self.poisoned=False
        ordinary(self.root);ordinary(self.registry)

    @classmethod
    def create(cls,root,registry,plan_path,plan_sha,cache,campaign_id,limits=None,protected_roots=()):
        plan,targets,metadata=inputs(plan_path,plan_sha,cache)
        limits=dict(DEFAULT_LIMITS if limits is None else limits)
        if set(limits)!=set(DEFAULT_LIMITS) or any(type(v) is not int or v<=0 or v>DEFAULT_LIMITS[k] for k,v in limits.items()):raise Stop('Limits may only be positive and no higher than reviewed proposals')
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}',campaign_id):raise Stop('Invalid campaign ID')
        c=cls(root,registry);cache=Path(cache).resolve();repo=Path(__file__).resolve().parents[1]
        protected=[cache,repo,*[Path(p).resolve() for p in protected_roots]]
        for destination in (c.root.resolve(),c.registry.resolve()):
            for source in protected:
                if destination==source or source in destination.parents or destination in source.parents:raise Stop('Protected input/repository overlap')
        if c.root==c.registry or c.root in c.registry.parents or c.registry in c.root.parents:raise Stop('Registry/campaign must be disjoint')
        binding=hash_bytes(encoded(dict(source_generation=GENERATION,targets=targets)))
        c.entry=c.registry/binding
        if c.root.exists():raise Stop('Fresh campaign root required')
        c.registry.mkdir(parents=True,exist_ok=True)
        # Permanent claim, including failed initializations: no silent campaign reset.
        c.entry.mkdir(exist_ok=False)
        c.root.mkdir(parents=True,exist_ok=False)
        c.contract=dict(schema=SCHEMA,campaign_id=campaign_id,created_utc=datetime.now(timezone.utc).isoformat(),
            plan_id=plan['plan_id'],plan_sha256=plan_sha,targets=targets,metadata_sha256=metadata,
            cache_root=str(cache),cache_verification='archives/train_meta.json and test_meta.json; image bytes NOT_VERIFIED',
            protected_roots=[str(p) for p in protected],
            limits=limits,registry_binding=binding,registry_root=str(c.registry.resolve()),
            execution_gate='OFFLINE_SAFETY_ONLY_NOT_REAL_DATA_APPROVAL')
        selected={str(r['image_id']):r for r in plan['plan']['pool'] if str(r['image_id']) in targets}
        c.contract['source_files']={i:dict(rgb=f"extracted/dataset/holdout/images/{selected[i]['oam_id']}_{i}.tif",
            annotation=f"extracted/dataset/holdout/masks/{selected[i]['oam_id']}_{i}.png",byte_validation='NOT_VERIFIED_R1B2B_REQUIRED') for i in targets}
        raw=encoded(c.contract)
        c._contract_sha=hash_bytes(raw)
        if len(raw)*2+CONTROL_MARGIN>limits['storage_bytes'] or any(shutil.disk_usage(p).free<len(raw)*2+CONTROL_MARGIN for p in (c.root,c.entry)):raise Stop('Initialization storage/free-space budget')
        durable(c.root/'contract.json',raw)
        durable(c.entry/'anchor.json',encoded(dict(root=str(c.root.resolve()),contract_sha256=hash_bytes(raw))))
        c.state=dict(operations={},fatal=None)
        c.sequence=0;c.previous=None
        (c.root/'events').mkdir();(c.root/'artifacts').mkdir()
        c._persist('CREATE')
        return c.root

    def _load(self,plan_path,plan_sha,expected_id=None):
        raw=(self.root/'contract.json').read_bytes();self.contract=json.loads(raw)
        if self.contract.get('schema')!=SCHEMA or self.contract['registry_root']!=str(self.registry.resolve()):raise Stop('Contract/registry mismatch')
        self.entry=self.registry/self.contract['registry_binding'];ordinary(self.entry)
        for p in self.entry.iterdir():
            ordinary(p)
            if p.name not in {'anchor.json','head.json','reconcile.lock'}:raise Stop('Unresolved registry publication')
        anchor=json.loads((self.entry/'anchor.json').read_bytes())
        if anchor!=dict(root=str(self.root.resolve()),contract_sha256=hash_bytes(raw)):raise Stop('Registry anchor/contract SHA mismatch')
        self._contract_sha=hash_bytes(raw)
        plan,targets,metadata=inputs(plan_path,plan_sha,Path(self.contract['cache_root']),expected_id)
        if (plan['plan_id']!=self.contract['plan_id'] or plan_sha!=self.contract['plan_sha256'] or targets!=self.contract['targets'] or metadata!=self.contract['metadata_sha256']):raise Stop('Fixed campaign input mismatch')
        head=json.loads((self.entry/'head.json').read_bytes())
        files=sorted((self.root/'events').iterdir());prev=None;state=dict(operations={},fatal=None)
        if len(files)!=head['sequence']:raise Stop('Orphan/missing journal record: STOP')
        for seq,path in enumerate(files,1):
            ordinary(path)
            if path.name!=f'{seq:08d}.json':raise Stop('Unexpected journal filename')
            data=path.read_bytes();event=json.loads(data)
            if event['sequence']!=seq or event['previous_sha256']!=prev or event['contract_sha256']!=hash_bytes(raw):raise Stop('Journal chain/contract mismatch')
            state['operations'].update(event['operations']);state['fatal']=event['fatal']
            prev=hash_bytes(data)
        if not files or prev!=head['sha256']:raise Stop('Journal head SHA mismatch')
        self.sequence=len(files);self.previous=prev;self.state=state;self._persisted_state=copy.deepcopy(state)
        self._audit()

    def _paths(self,op):
        return self.root/op['partial_path'],self.root/op['output_path']

    def _audit(self):
        expected={self.root/'contract.json',*list((self.root/'events').iterdir())}
        for op in self.state['operations'].values():
            partial,final=self._paths(op)
            for path in (partial,final):ordinary(path)
            if op['status']=='SUCCESS':
                if partial.exists() or not final.is_file() or final.stat().st_size!=op['file']['bytes'] or sha(final)!=op['file']['sha256']:raise Stop('Completed artifact SHA/size/existence mismatch')
                expected.add(final)
            else:
                for path in (partial,final):
                    if path.exists():
                        if path.stat().st_size>op['max_output_bytes']:raise Stop('Partial file exceeds reservation')
                        expected.add(path)
                if op['status'] in TERMINAL:
                    for receipt in op.get('retained_files',[]):
                        p=self.root/receipt['path']
                        if not p.is_file() or p.stat().st_size!=receipt['bytes'] or sha(p)!=receipt['sha256']:raise Stop('Retained partial SHA mismatch')
        for p in (self.root/'recovered-locks').glob('*') if (self.root/'recovered-locks').exists() else []:
            ordinary(p)
            if not p.is_file() or sha(p)!=p.stem:raise Stop('Archived lock SHA mismatch')
            expected.add(p)
        if (self.root/'campaign.lock').exists():expected.add(self.root/'campaign.lock')
        actual=set()
        for p in self.root.rglob('*'):
            ordinary(p)
            if p.is_file():actual.add(p)
        if actual!=expected:raise Stop('Unowned/missing campaign files: STOP')

    def usage(self):
        ops=list(self.state['operations'].values())
        disk=sum(p.stat().st_size for base in (self.root,self.entry) for p in base.rglob('*') if p.is_file())
        held=0
        for op in ops:
            if op['status'] not in TERMINAL:
                present=sum(p.stat().st_size for p in self._paths(op) if p.is_file())
                held+=max(0,op['reserved_storage_bytes']-present)
        return dict(network_charged_bytes=sum(o.get('charged_network_bytes',o['reserved_network_bytes']) for o in ops),
                    known_received_bytes=sum(o['received_bytes'] for o in ops),
                    requests=sum(o['kind']=='stac' for o in ops),disk_bytes=disk,unresolved_storage_reservation_bytes=held,
                    storage_including_reservations_bytes=disk+held,
                    unresolved_operations=[o['operation_id'] for o in ops if o['status'] not in TERMINAL])

    def _space(self,extra,control=False):
        u=self.usage();limit=self.contract['limits']['storage_bytes']
        need=u['disk_bytes']+u['unresolved_storage_reservation_bytes']+extra+(0 if control else CONTROL_MARGIN)
        if need>limit or any(shutil.disk_usage(p).free<u['unresolved_storage_reservation_bytes']+extra+CONTROL_MARGIN for p in (self.root,self.entry)):
            raise Stop('Storage/free-space budget would be exceeded')

    def _persist(self,reason):
        prior=getattr(self,'_persisted_state',dict(operations={}))
        changed={k:copy.deepcopy(v) for k,v in self.state['operations'].items() if prior['operations'].get(k)!=v}
        event=dict(sequence=self.sequence+1,previous_sha256=self.previous,contract_sha256=sha(self.root/'contract.json'),
                   utc=datetime.now(timezone.utc).isoformat(),reason=reason,operations=changed,fatal=self.state['fatal'])
        data=encoded(event);head=encoded(dict(sequence=event['sequence'],sha256=hash_bytes(data)))
        self._space(len(data)+len(head),control=True)
        try:
            durable(self.root/'events'/f"{event['sequence']:08d}.json",data)
            temporary=self.entry/'head.pending'
            durable(temporary,head);os.replace(temporary,self.entry/'head.json')
            self.sequence=event['sequence'];self.previous=hash_bytes(data)
            self._persisted_state=copy.deepcopy(self.state)
        except BaseException:
            self.poisoned=True;raise

    @classmethod
    def open(cls,root,registry,plan_path,plan_sha,expected_id=None):
        c=cls(root,registry);c._load(plan_path,plan_sha,expected_id)
        if (c.entry/'reconcile.lock').exists():raise Stop('Stale reconciliation guard: STOP')
        if c.state['fatal'] or c.usage()['unresolved_operations']:raise Stop('Unresolved operations/fatal state require inspection and explicit reconciliation')
        c._space(512)
        durable(c.root/'campaign.lock',encoded(dict(pid=os.getpid(),nonce=uuid.uuid4().hex)))
        c._lock_sha=sha(c.root/'campaign.lock')
        c.closed=False
        return c

    def __enter__(self):return self

    def __exit__(self,kind,value,tb):
        unresolved=self.usage()['unresolved_operations']
        if not self.poisoned and (kind is not None or not unresolved):
            if not (self.root/'campaign.lock').is_file() or sha(self.root/'campaign.lock')!=self._lock_sha:raise Stop('Owner lock changed: retained')
            (self.root/'campaign.lock').unlink();self.closed=True
        if kind is None and unresolved:raise Stop('Pending operation: lock retained; explicit reconciliation required')

    def _ready(self):
        if self.closed or self.poisoned or self.state['fatal'] or any(o['status']=='INTERRUPTED' for o in self.state['operations'].values()):raise Stop('Campaign not available: STOP/reconcile')
        if not (self.root/'campaign.lock').is_file() or sha(self.root/'campaign.lock')!=self._lock_sha:raise Stop('Owner lock changed: STOP')
        if hash_bytes(encoded(self.contract))!=self._contract_sha or sha(self.root/'contract.json')!=self._contract_sha:
            raise Stop('Fixed contract changed in memory/on disk')
        if self.state!=self._persisted_state:raise Stop('Unjournaled state modification: STOP')

    def reserve(self,operation_id,image_id,split,kind,path,max_output_bytes,network_bytes=0):
        self._ready()
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}',operation_id) or operation_id in self.state['operations']:raise Stop('Invalid/duplicate operation ID')
        if type(image_id) is not int or self.contract['targets'].get(str(image_id))!=split:raise Stop('Target image/split outside fixed cohort')
        if kind not in KINDS or type(max_output_bytes) is not int or max_output_bytes<=0 or type(network_bytes) is not int:raise Stop('Invalid operation')
        limits=self.contract['limits'];u=self.usage()
        if kind=='stac':
            if not 0<network_bytes<=limits['response_bytes'] or u['requests']+1>limits['requests']:raise Stop('Response/request budget')
        elif network_bytes!=0:raise Stop('Image/annotation/Range network is prohibited')
        if network_bytes<0 or u['network_charged_bytes']+network_bytes>limits['network_bytes']:raise Stop('Network budget')
        relative(path);final=self.root/path;ordinary(final)
        if final.exists() or any(o['output_path'].casefold()==path.casefold() for o in self.state['operations'].values()):raise Stop('Existing/duplicate output path')
        op=dict(operation_id=operation_id,image_id=image_id,split=split,kind=kind,output_path=path,
                partial_path=f'artifacts/{operation_id}.partial',max_output_bytes=max_output_bytes,
                reserved_storage_bytes=2*max_output_bytes,reserved_network_bytes=network_bytes,
                received_bytes=0,network_complete=kind!='stac',status='RESERVED')
        relative(op['partial_path'])
        occupied={p.casefold() for o in self.state['operations'].values() for p in (o['output_path'],o['partial_path'])}
        if op['partial_path'].casefold()==path.casefold() or any(p.casefold() in occupied for p in (path,op['partial_path'])):raise Stop('Output/partial path collision')
        if (self.root/op['partial_path']).exists():raise Stop('Partial output exists')
        self.state['operations'][operation_id]=op
        try:self._space(16384);self._persist('RESERVE')
        except BaseException:
            self.state['operations'].pop(operation_id);raise

    def receive(self,operation_id,reader):
        self._ready();op=self._operation(operation_id)
        if op['kind']!='stac' or op['received_bytes'] or op['network_complete']:raise Stop('Network operation already started/not STAC')
        data=bytearray();self._space(16384)
        try:
            while len(data)<op['reserved_network_bytes']:
                remaining=op['reserved_network_bytes']-len(data)
                chunk=reader(remaining)  # adapter must never over-read; no HTTP client in this PR
                if not isinstance(chunk,bytes):raise Stop('Reader must return bounded bytes')
                data.extend(chunk);op['received_bytes']=len(data)
                if len(data)>op['reserved_network_bytes']:
                    self.state['fatal']='READER_OVERRUN';op['charged_network_bytes']=len(data);raise Stop('Reader exceeded reserved response')
                self._persist('RECEIVE_PROGRESS')
                if not chunk:break
            if len(data)==op['reserved_network_bytes']:raise Stop('Response bound reached without verified EOF')
            op['network_complete']=True;self._persist('RECEIVE_END')
            return bytes(data)
        except BaseException:
            op['status']='INTERRUPTED';self._persist('RECEIVE_UNCERTAIN');raise

    def _operation(self,operation_id):
        op=self.state['operations'][operation_id]
        if op['status'] in TERMINAL:raise Stop('Operation already settled')
        return op

    def write_bytes(self,operation_id,data):
        self._ready();op=self._operation(operation_id)
        if not isinstance(data,bytes) or len(data)>op['max_output_bytes']:raise Stop('Output exceeds reservation')
        self._space(16384);partial,_=self._paths(op)
        try:
            durable(partial,data)
            receipt=dict(bytes=partial.stat().st_size,sha256=sha(partial))
            if receipt!=dict(bytes=len(data),sha256=hash_bytes(data)):raise Stop('Written partial differs from supplied bytes')
            op.update(status='WRITTEN',written_file=receipt);self._persist('WRITE_COMPLETE')
        except BaseException:
            op['status']='INTERRUPTED'
            if not self.poisoned:self._persist('WRITE_INTERRUPTED')
            raise

    def commit(self,operation_id):
        self._ready();op=self._operation(operation_id)
        if op['status']!='WRITTEN':raise Stop('Commit requires journaled WRITTEN state')
        if not op['network_complete']:raise Stop('Network consumption unresolved')
        partial,final=self._paths(op)
        try:
            ordinary(partial)
            if (not partial.is_file() or partial.stat().st_size>op['max_output_bytes'] or
                    dict(bytes=partial.stat().st_size,sha256=sha(partial))!=op.get('written_file')):
                raise Stop('Written partial size/SHA mismatch before commit')
            self._space(16384);ordinary(final);final.parent.mkdir(parents=True,exist_ok=True)
            os.link(partial,final)  # exclusive no-overwrite publication; retain reservation for both names
            receipt=dict(bytes=final.stat().st_size,sha256=sha(final))
            if receipt!=op['written_file']:raise Stop('Published file differs from written receipt')
            partial.unlink()
            op.update(status='SUCCESS',charged_network_bytes=op['received_bytes'],file=receipt)
            self._persist('COMMIT')
        except BaseException:
            op['status']='INTERRUPTED'
            if not self.poisoned:self._persist('PUBLISH_INTERRUPTED')
            raise

    @classmethod
    def inspect(cls,root,registry,plan_path,plan_sha,expected_id=None):
        c=cls(root,registry);c._load(plan_path,plan_sha,expected_id)
        reasons=[]
        if c.usage()['unresolved_operations']:reasons.append('UNRESOLVED_OPERATIONS')
        if (c.root/'campaign.lock').exists():reasons.append('CAMPAIGN_OWNER_LOCK_PRESENT')
        if (c.entry/'reconcile.lock').exists():reasons.append('RECONCILIATION_LOCK_PRESENT')
        if c.state['fatal']:reasons.append('FATAL_ACCOUNTING_STATE')
        return dict(contract=c.contract,usage=c.usage(),operations=c.state['operations'],
                    partial_files=[dict(operation_id=o['operation_id'],path=str(p.relative_to(c.root)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for o in c.state['operations'].values() if o['status'] not in TERMINAL for p in c._paths(o) if p.is_file()],
                    lock_sha256=sha(c.root/'campaign.lock') if (c.root/'campaign.lock').exists() else None,
                    reconcile_lock_sha256=sha(c.entry/'reconcile.lock') if (c.entry/'reconcile.lock').exists() else None,
                    stop_reasons=reasons,status='STOP' if reasons else 'CLEAN_PAUSED')

    @classmethod
    def reconcile(cls,root,registry,plan_path,plan_sha,*,acknowledge_stopped_owner=False,expected_lock_sha=None,expected_id=None):
        if not acknowledge_stopped_owner:raise Stop('Explicit stopped-owner acknowledgement required')
        c=cls(root,registry);c._load(plan_path,plan_sha,expected_id)
        if c.state['fatal']:raise Stop('Fatal accounting breach requires independent review')
        guard=c.entry/'reconcile.lock';c._space(16384);durable(guard,b'operator reconciliation\n')
        try:
            lock=c.root/'campaign.lock'
            if (sha(lock) if lock.exists() else None)!=expected_lock_sha:raise Stop('Lock SHA changed/mismatched')
            c._space(16384)
            if lock.exists():
                folder=c.root/'recovered-locks';folder.mkdir(exist_ok=True)
                os.link(lock,folder/(expected_lock_sha+'.json'));lock.unlink()
            durable(lock,encoded(dict(pid=os.getpid(),nonce=uuid.uuid4().hex,reconciliation=True)))
            for op in c.state['operations'].values():
                if op['status'] not in TERMINAL:
                    op.update(status='RECONCILED_FULL_CHARGE',charged_network_bytes=op.get('charged_network_bytes',op['reserved_network_bytes']),
                              retained_files=[dict(path=str(p.relative_to(c.root)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in c._paths(op) if p.is_file()])
            c._persist('OPERATOR_RECONCILE_FULL_CHARGE');lock.unlink()
        finally:guard.unlink()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','inspect','resume-check','reconcile'])
    for name in ('root','registry','plan','cache'):p.add_argument('--'+name,type=Path,required=name!='cache')
    p.add_argument('--plan-sha256',required=True);p.add_argument('--plan-id');p.add_argument('--campaign-id')
    p.add_argument('--protect-root',type=Path,action='append',default=[])
    p.add_argument('--acknowledge-stopped-owner',action='store_true');p.add_argument('--lock-sha256')
    a=p.parse_args(argv)
    if a.action=='init':
        if not a.cache or not a.campaign_id:p.error('init requires --cache and --campaign-id')
        if a.plan_id:inputs(a.plan,a.plan_sha256,a.cache,a.plan_id)
        Campaign.create(a.root,a.registry,a.plan,a.plan_sha256,a.cache,a.campaign_id,protected_roots=a.protect_root)
    elif a.action=='reconcile':Campaign.reconcile(a.root,a.registry,a.plan,a.plan_sha256,acknowledge_stopped_owner=a.acknowledge_stopped_owner,expected_lock_sha=a.lock_sha256,expected_id=a.plan_id)
    elif a.action=='resume-check':
        with Campaign.open(a.root,a.registry,a.plan,a.plan_sha256,a.plan_id):pass
    result=Campaign.inspect(a.root,a.registry,a.plan,a.plan_sha256,a.plan_id);print(json.dumps(result,indent=2));return 0 if result['status']=='CLEAN_PAUSED' else 2


if __name__=='__main__':raise SystemExit(main())
