"""Opt-in Zenodo10 CUDA Train7 smoke: fixed 20-step schedule, checkpoint10/20."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace
import uuid

import m1_7_dataset as data
import m1_7_g1b_gpu_smoke as shared

KIND='OUEM_M1_7_G2_ZENODO10_BOUNDED20'
TRAIN_IDS=sorted(i for i,r in data.ROLES.items() if r=='train')


def config(seed):
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Seed out of range')
    c=dict(shared.CONFIG,seed=seed,steps=20,crop_policy='uniform-integer-valid-window',schedule_version=1)
    c.pop('crop_origin');return c


def digest(value):return hashlib.sha256(data.encoded(value)).hexdigest()


def make_schedule(manifest,seed=0):
    data.check_manifest(manifest);config(seed)
    samples={s['image_id']:s for s in manifest['samples']};rng=random.Random(seed);schedule=[]
    for cycle in range(3):
        ids=list(TRAIN_IDS);rng.shuffle(ids)
        for rid in ids:
            if len(schedule)==20:break
            s=samples[rid];h,w=s['output_shape'];size=224
            if h<size or w<size:raise ValueError('Scene smaller than patch')
            schedule.append(dict(step=len(schedule)+1,cycle=cycle,image_id=rid,oam_id=s['oam_id'],
                                 crop_xywh=[rng.randint(0,w-size),rng.randint(0,h-size),size,size]))
    validate_schedule(schedule,manifest);return schedule


def validate_schedule(schedule,manifest):
    if len(schedule)!=20:raise ValueError('Exactly 20 schedule entries required')
    samples={s['image_id']:s for s in manifest['samples']}
    for index,item in enumerate(schedule):
        rid=item['image_id'];crop=item['crop_xywh']
        if any(type(item[k]) is not int for k in ['step','cycle','image_id']):raise ValueError('Integer schedule identity required')
        if rid not in TRAIN_IDS or item['step']!=index+1 or item['cycle']!=index//7:raise ValueError('Schedule step/Train identity invalid')
        s=samples[rid];h,w=s['output_shape']
        if s['split']!='train' or item['oam_id']!=s['oam_id']:raise ValueError('Schedule source/role mismatch')
        if len(crop)!=4 or any(type(v) is not int for v in crop):raise ValueError('Integer crop required')
        x,y,cw,ch=crop
        if cw!=224 or ch!=224 or x<0 or y<0 or x+cw>w or y+ch>h:raise ValueError('Out-of-range crop/padding forbidden')
    for start in [0,7,14]:
        ids=[s['image_id'] for s in schedule[start:start+7]]
        if len(set(ids))!=len(ids):raise ValueError('Repeated scene within cycle')
    if set(Counter(s['image_id'] for s in schedule).values())!={2,3}:raise ValueError('Scene counts must be 2 or 3')


def inputs(manifest):
    return dict(dataset_id=manifest['dataset_id'],source_generation=manifest['source_generation'],manifest_sha256=data.MANIFEST_SHA256,
                product_hashes={str(s['image_id']):s['output_hashes'] for s in manifest['samples']},train_ids=TRAIN_IDS)


def compatible(state,manifest,cfg,initializer,environment,git_sha,allow_change=False):
    if state.get('kind')!=KIND or state.get('config')!=cfg or state.get('inputs')!=inputs(manifest) or state.get('initializer')!=initializer:
        raise ValueError('Checkpoint kind/config/data/initializer mismatch (G1-B/HF forbidden)')
    n=state.get('completed_steps')
    if type(n) is not int or n not in (10,20) or state.get('next_step')!=n+1 or state.get('remaining_steps')!=list(range(n+1,21)) or state.get('executed_steps')!=list(range(1,n+1)):
        raise ValueError('Checkpoint completed/remaining step mismatch')
    validate_schedule(state['schedule'],manifest)
    if digest(state['schedule'])!=state['schedule_sha256']:raise ValueError('Schedule fingerprint mismatch')
    history=state['history']
    if len(history)!=n:raise ValueError('History length mismatch')
    for i,h in enumerate(history):
        if any(h.get(k)!=state['schedule'][i][k] for k in ['step','image_id','oam_id','crop_xywh']):raise ValueError('Executed history/schedule mismatch')
        if not isinstance(h.get('step_loss'),(int,float)) or not math.isfinite(h['step_loss']):raise ValueError('Invalid history loss')
    changes={k:[state['environment'].get(k),v] for k,v in environment.items() if state['environment'].get(k)!=v}
    if state['git_sha']!=git_sha:changes['git_sha']=[state['git_sha'],git_sha]
    if changes and not allow_change:raise ValueError('Environment/commit changed; explicit override required')
    return changes


def write_new(path,value):
    with path.open('xb') as stream:stream.write(data.encoded(value))


def claim_continuation(backup,checkpoint_sha256):
    """Both uninterrupted and resumed step11 must consume this same persistent token."""
    import re
    if not isinstance(checkpoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',checkpoint_sha256):raise ValueError('Canonical checkpoint SHA required')
    path=backup/('step11-from-'+checkpoint_sha256+'.claim.json')
    write_new(path,dict(checkpoint_sha256=checkpoint_sha256,first_step=11,last_step=20,
                        utc=datetime.now(timezone.utc).isoformat(),policy='single-consumption; never automatically release'))
    return path.name


def open_lineage(args,record,state=None):
    if state:
        data.require_hash(args.backup/'lineage.json',state['lineage_sha256'])
        lineage=json.loads((args.backup/'lineage.json').read_text())
        if lineage['run_id']!=args.run_id or lineage['config']!=record['config'] or lineage['inputs']!=record['inputs'] or lineage['schedule_sha256']!=record['schedule_sha256']:
            raise ValueError('Persistent lineage identity mismatch')
        receipt=json.loads((args.backup/f"step{state['completed_steps']:02d}"/'checkpoint.json').read_text())
        if receipt['sha256']!=args.resume_sha256:raise ValueError('Resume SHA differs from backed-up checkpoint receipt')
    else:
        data.fresh_external(args.backup,[args.data,args.checkpoint,args.g1a_result,args.output])
        lineage=dict(kind=KIND,run_id=args.run_id,lineage_uuid=uuid.uuid4().hex,config=record['config'],inputs=record['inputs'],
                     initializer=record['initializer'],schedule_sha256=record['schedule_sha256'])
        write_new(args.backup/'lineage.json',lineage)
    record['lineage_sha256']=data.sha(args.backup/'lineage.json')
    segment=args.backup/'segments'/record['segment_id'];segment.mkdir(parents=True,exist_ok=False)
    record['backup_segment']=str(segment)


def save_checkpoint(args,record,model,optimizer,torch):
    n=record['completed_steps'];record['stage']=f'checkpoint_save_{n}'
    payload=dict(kind=KIND,run_id=args.run_id,config=record['config'],inputs=record['inputs'],initializer=record['initializer'],
        git_sha=record['git_sha'],environment=record['environment'],completed_steps=n,next_step=n+1,executed_steps=list(range(1,n+1)),remaining_steps=list(range(n+1,21)),
        schedule=record['schedule'],schedule_sha256=record['schedule_sha256'],history=record['history'],
        lineage_sha256=record['lineage_sha256'],model=model.state_dict(),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
        fingerprints=shared.fingerprints(model,optimizer,torch))
    path=args.output/f'zenodo10_g2_step{n:02d}.pth'
    sha,seconds=shared.timed_cuda(torch,lambda:shared.save_state(path,payload))
    receipt=dict(step=n,filename=path.name,sha256=sha,bytes=path.stat().st_size,save_seconds=seconds,backup='NOT_VERIFIED',fingerprints=payload['fingerprints'])
    record['checkpoints'].append(receipt)
    record['stage']=f'checkpoint_backup_{n}';started=time.perf_counter()
    receipt['backup']=shared.copy_backup(path,sha,args.backup/f'step{n:02d}')
    receipt['backup_seconds']=time.perf_counter()-started
    write_new(args.backup/f'step{n:02d}'/'checkpoint.json',receipt)
    return sha


def run(args,record):
    import torch
    # Reuse the G1-B CUDA/import/data40/initializer gates; never invoke its training/restore branch.
    shared.run(SimpleNamespace(data=args.data,checkpoint=args.checkpoint,g1a_result=args.g1a_result,image_id=3,preflight_only=True),record)
    record['status']='NOT_VERIFIED';manifest=json.loads((args.data/'manifest.json').read_text())
    record['inputs']=inputs(manifest)
    state=None
    if args.resume:
        record['stage']='checkpoint_compatibility';state=shared.safe_load(args.resume,args.resume_sha256)
        record['environment_changes']=compatible(state,manifest,record['config'],record['initializer'],record['environment'],record['git_sha'],args.allow_environment_change)
        if state['run_id']!=args.run_id:raise ValueError('Run ID mismatch')
        if not args.restore_only and (state['completed_steps']!=10 or args.stop_after!=20):raise ValueError('Training resume is step10 -> 11..20 only')
        # Stored schedule is authoritative; never re-draw the remaining crops after restoring RNG.
        record.update(schedule=state['schedule'],schedule_sha256=state['schedule_sha256'],history=state['history'],completed_steps=state['completed_steps'])
        record['resume']=dict(checkpoint_sha256=args.resume_sha256,completed_steps=state['completed_steps'],next_step=state['next_step'])
    else:
        record['schedule']=make_schedule(manifest,args.seed);record['schedule_sha256']=digest(record['schedule'])
    if args.preflight_only:
        record['status']='G2_PREFLIGHT_PASS_TRAINING_NOT_RUN';return
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed)
    record['stage']='model_construction';torch.cuda.reset_peak_memory_stats()
    import satlaspretrain_models as satlas
    weights=None if state else shared.safe_load(args.checkpoint,record['initializer']['sha256'])
    if weights is not None and (not isinstance(weights,dict) or not weights or not all(isinstance(v,torch.Tensor) for v in weights.values())):
        raise ValueError('Official initializer must contain flat tensors')
    started=time.perf_counter()
    model=satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights)
    del weights
    model=model.to(device='cuda',dtype=torch.float32);model.requires_grad_(True)
    c=record['config'];optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'],betas=tuple(c['betas']),eps=c['eps'])
    torch.cuda.synchronize();record['construction_seconds']=time.perf_counter()-started
    record['memory_after_construction']=shared.cuda_memory(torch)
    record['initialization']='strict G2 trained-state restore' if state else 'official Backbone/FPN; fresh Upsample/SEGMENT2'
    if state:
        record['stage']='state_restore'
        record['restore_checks'],record['restore_seconds']=shared.timed_cuda(torch,lambda:shared.restore(model,optimizer,state,torch))
        record['memory_after_restore']=shared.cuda_memory(torch)
    open_lineage(args,record,state)
    if args.restore_only:
        record.update(status='G2_STATE_RESTORE_PASS_NO_TRAINING',continuation_determinism='NOT_VERIFIED: state comparison only');return
    start=record['completed_steps']+1;checkpoint10=args.resume_sha256 if state else None
    del state
    for number in range(start,args.stop_after+1):
        item=record['schedule'][number-1];entry=dict(item);record['active_step']=entry;record['stage']=f'load_patch_{number}'
        started=time.perf_counter()
        x,y,_,meta=shared.load_patch(args.data,args.data/'manifest.json',item['image_id'],'train',item['crop_xywh'])
        entry.update(positive_fraction=meta['positive_fraction'],input=meta,load_seconds=time.perf_counter()-started)
        if number==11:
            record['stage']='claim_step11';record['continuation_claim']=claim_continuation(args.backup,checkpoint10)
        record['stage']='training_step';x=x.to('cuda');y=y.to('cuda')
        shared.one_step(model,optimizer,x,y,torch,entry,evaluate_before=False)
        record['nan_inf_detected']=entry['nan_inf_detected']
        entry.pop('completed_steps',None);record['completed_steps']=number;record['history'].append(entry);record.pop('active_step',None)
        record['scene_counts']=dict(Counter(str(s['image_id']) for s in record['history']))
        record['stage']=f'persist_step_{number}'
        write_new(args.output/f'step_{number:02d}.json',entry)
        write_new(Path(record['backup_segment'])/f'step_{number:02d}.json',entry)
        print(json.dumps(dict(step=number,image_id=item['image_id'],loss=entry['step_loss'])),flush=True)
        del x,y
        if number in (10,20):
            h=save_checkpoint(args,record,model,optimizer,torch)
            if number==10:checkpoint10=h
    record.update(status='G2_BOUNDED_STEPS_PASS_RESTORE_PENDING',memory_final=shared.cuda_memory(torch),
                  remaining_steps=list(range(record['completed_steps']+1,21)),continuation_determinism='GPU_NOT_VERIFIED; CPU surrogate tested separately')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','checkpoint','g1a-result','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--backup',type=Path);p.add_argument('--stop-after',type=int,choices=[10,20],default=20)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--resume-sha256')
    p.add_argument('--restore-only',action='store_true');p.add_argument('--allow-environment-change',action='store_true')
    return p


def reserve(args):
    import re
    config(args.seed)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',args.run_id):raise ValueError('Safe run ID required')
    if bool(args.resume)!=bool(args.resume_sha256) or args.restore_only and not args.resume:raise ValueError('Resume path/SHA required together')
    if args.preflight_only and (args.resume or args.restore_only or args.backup):raise ValueError('Preflight has no resume/backup')
    if args.allow_environment_change and not args.resume:raise ValueError('Environment override is resume-only')
    if not args.preflight_only and not args.backup:raise ValueError('Training/restore requires persistent lineage --backup')
    protected=[args.data,args.checkpoint,args.g1a_result]+([args.resume] if args.resume else [])
    if args.backup:
        b=args.backup.resolve()
        if b==data.REPO or data.REPO in b.parents or any(b==s.resolve() or s.resolve() in b.parents or b in s.resolve().parents for s in protected[:3]+[args.output]):
            raise ValueError('Backup overlaps inputs/repository/output')
        if args.resume:
            if not args.backup.is_dir():raise ValueError('Resume requires the original persistent lineage directory')
        elif args.backup.exists() or args.backup.is_symlink():raise ValueError('Fresh backup required')
        protected.append(args.backup)
    data.fresh_external(args.output,protected)


def main(argv=None):
    args=parser().parse_args(argv)
    try:reserve(args)
    except (ValueError,OSError) as exc:parser().error(str(exc))
    record=dict(status='NOT_VERIFIED',stage='startup',run_id=args.run_id,segment_id=uuid.uuid4().hex,config=config(args.seed),
                utc=datetime.now(timezone.utc).isoformat(),completed_steps=0,history=[],checkpoints=[],scene_counts={},memory_final='NOT_MEASURED',restore_checks='NOT_MEASURED',nan_inf_detected='NOT_MEASURED')
    started=time.perf_counter();code=0
    try:run(args,record)
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='CUDA_NOT_VERIFIED' if record['status']=='CUDA_NOT_VERIFIED' else 'FAIL_STOPPED'
        record['exception']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());code=1
        active=record.get('active_step',{})
        if isinstance(active.get('nan_inf_detected'),bool):record['nan_inf_detected']=active['nan_inf_detected']
        if 'nonfinite' in str(exc).lower():
            record['nan_inf_detected']=True
            if active:active['nan_inf_detected']=True
        try:
            import torch
            if torch.cuda.is_available():record['memory_at_failure']=shared.cuda_memory(torch)
        except Exception:pass
    record['total_seconds']=time.perf_counter()-started
    record['executed_steps']=[s['step'] for s in record['history']]
    record['remaining_steps']=list(range(record['completed_steps']+1,21))
    record['scene_counts']=dict(Counter(str(s['image_id']) for s in record['history']))
    write_new(args.output/'result.json',record)
    if 'backup_segment' in record:
        try:
            path=Path(record['backup_segment'])/'result.json'
            with (args.output/'result.json').open('rb') as src,path.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src,dst)
            data.require_hash(path,data.sha(args.output/'result.json'))
        except Exception as exc:
            record.update(status='FAIL_STOPPED',result_backup_error=str(exc));code=1
            (args.output/'result.json').write_bytes(data.encoded(record))  # only this process's newly reserved result
    print(json.dumps(dict(status=record['status'],run_id=args.run_id,completed_steps=record['completed_steps'],
        scene_counts=record['scene_counts'],stage=record['stage'],result_json=str(args.output/'result.json'),
        checkpoint_sha256={str(c['step']):c['sha256'] for c in record['checkpoints']},
        error=record.get('exception',{}).get('message')),indent=2));return code


if __name__=='__main__':raise SystemExit(main())
