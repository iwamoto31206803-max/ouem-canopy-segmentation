"""D6 segmented FP32 training. CLI requires one GPU; tests inject a CPU model."""
import argparse
import math
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from d6a3_produce import canonical, identifier, need, pinned, plain, write_new
from d6a_research_qa import sha, strict_json
from d6b_bundle import read_bundle
import d6b_relay as relay
from m1_4_bounded_overfit import finite_parameters, fingerprint
from m1_6_colab_gpu_smoke import cuda_memory
from m1_7_g1b_gpu_smoke import finite_optimizer
from m1_7_g3a_eval import comparison
from r2_train import _checkpoint, make_plan
import research_cuda as d5
from research_val18_recovery import verified_load

KIND='OUEM_D6_SEGMENTED_V1'
PLAN='ouem-d6-run-v1'
REPO=Path(__file__).resolve().parents[1]


def json_new(path, value):
    write_new(path,canonical(value),8*1024**2)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def valid_id(value):
    return isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value)


def make_binding(spec, bundle_pin, data, historical, *, run_id, total, save_every, eval_every):
    need(valid_id(run_id) and all(type(n) is int and 0<n<=10000 for n in (total,save_every,eval_every))
         and total>=save_every and total>=eval_every and eval_every%save_every==0,
         'Invalid run/cadence; evaluation must fall on checkpoint multiples')
    schedule=make_plan(data,seed=0,steps=total)
    schedule['historical_val']=historical.entries
    binding=dict(schema=PLAN,kind=KIND,run_id=run_id,commit=spec['commit'],
                 bundle_sha256=bundle_pin,initializer_sha256=spec['initializer_sha256'],
                 training_data=data.identity,historical_val=historical.identity,
                 schedule_sha256=identifier(schedule),
                 config=dict(total=total,save_every=save_every,eval_every=eval_every,seed=0,
                             batch=1,precision='FP32',amp=False,tf32=False,scheduler=None,scaler=None,
                             optimizer='AdamW',adamw=dict(d5.ADAMW,betas=list(d5.ADAMW['betas']))))
    return binding,schedule


def rng(device):
    numpy=np.random.get_state()
    return dict(python=random.getstate(),numpy=(numpy[0],numpy[1].tolist(),*numpy[2:]),
                cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])


def set_rng(state,device):
    need(len(state['cuda'])==(torch.cuda.device_count() if device=='cuda' else 0),'CUDA RNG topology')
    random.setstate(state['python'])
    numpy=state['numpy']
    np.random.set_state((numpy[0],np.asarray(numpy[1],dtype=np.uint32),*numpy[2:]))
    torch.set_rng_state(state['cpu'])
    if device=='cuda':
        torch.cuda.set_rng_state_all(state['cuda'])


def state_pins(model,optimizer,random_state):
    return {k:fingerprint(v) for k,v in dict(model=model,optimizer=optimizer,rng=random_state).items()}


def snapshot(model,optimizer,device,binding,schedule,session,environment,parent,step):
    model_state,optim_state,random_state=model.state_dict(),optimizer.state_dict(),rng(device)
    return dict(kind=KIND,binding=binding,schedule=schedule,completed=step,
                next_item=next_item(schedule,step),session=session,environment=environment,parent=parent,
                model=model_state,optimizer=optim_state,rng=random_state,
                fingerprints=state_pins(model_state,optim_state,random_state))


def next_item(schedule,step):
    return schedule['train'][step] if step<len(schedule['train']) else None


def checked_restore(folder,pin,binding,schedule,session,environment,model,optimizer,device):
    receipt=relay.verify(folder,pin)
    details=receipt['details']
    need(details['kind']=='checkpoint' and details['run_id']==binding['run_id']
         and details['session']!=session and details['binding_sha256']==identifier(binding),'Resume receipt binding')
    member=receipt['files']['checkpoint.pth']
    path=folder/'checkpoint.pth'
    state=verified_load(path,member['sha256'],member['size'],torch)
    step=state['completed']
    need(state['kind']==KIND and state['binding']==binding and state['schedule']==schedule
         and type(step) is int and 0<step<=len(schedule['train']) and state['next_item']==next_item(schedule,step)
         and state['session']==details['session'] and step==details['completed']
         and state['parent']==details['parent'] and state['environment']==details['environment'],
         'Checkpoint binding or schedule mismatch')
    need(state['environment']['software']==environment['software'],'Runtime software changed across sessions')
    expected=state_pins(state['model'],state['optimizer'],state['rng'])
    need(expected==state['fingerprints'],'Checkpoint fingerprint mismatch')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    set_rng(state['rng'],device)
    need(state_pins(model.state_dict(),optimizer.state_dict(),rng(device))==expected,'Restored state fingerprint mismatch')
    finite_parameters(model)
    finite_optimizer(optimizer,torch)
    return dict(completed=step,session=state['session'],receipt_sha256=pin,
                checkpoint_sha256=member['sha256'],fingerprints=expected)


def validate_sessions(output,backup,protected):
    output,backup=map(plain,(output,backup))
    for dest,other in ((output,backup),(backup,output)):
        need(not dest.exists() and dest.parent.is_dir(),'New session output directory required')
        need(not any(dest.is_relative_to(p) or p.is_relative_to(dest)
                     for p in (other,REPO,*map(plain,protected))),'Session destination overlap')
    output.mkdir()
    backup.mkdir()
    return output,backup


def run_segment(data,historical,binding,schedule,model_factory,device,environment,
                session,output,backup,*,updates,seconds,reserve,max_vram,resume=None,
                receipt_sha=None,protected=(),clock=time.monotonic):
    """An append-only segment with explicit checkpoint commits at completed updates."""
    need(valid_id(session) and type(updates) is int and updates>0 and type(max_vram) is int and max_vram>0
         and all(type(v) in (int,float) and math.isfinite(v) for v in (seconds,reserve))
         and 0<reserve<seconds and bool(resume)==bool(receipt_sha),'Invalid segment limits')
    need(binding['schedule_sha256']==identifier(schedule)
         and binding['config']['total']==len(schedule['train']),'Run schedule mismatch')
    started=clock()
    deadline=started+seconds-reserve
    limit=binding['config']['total']
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device=='cuda':
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()
    model=model_factory().to(device=device,dtype=torch.float32).requires_grad_(True).train()
    optimizer=torch.optim.AdamW(model.parameters(),**binding['config']['adamw'])
    parent=checked_restore(resume,receipt_sha,binding,schedule,session,environment,
                           model,optimizer,device) if resume else None
    completed=parent['completed'] if parent else 0
    need(completed<limit,'Run already complete')
    output,backup=validate_sessions(output,backup,(*protected,*([resume] if resume else [])))
    published=[]
    pending=output/'pending'
    pending.mkdir()
    last=completed

    def resource():
        if device=='cuda':
            mem=cuda_memory(torch)
            need(torch.cuda.max_memory_reserved()<=max_vram,'Reserved CUDA memory exceeds ceiling')
            return mem
        return {}

    def push(kind,step):
        nonlocal pending
        name=f'{kind}-{step:04d}'
        details=dict(kind=kind,completed=step,run_id=binding['run_id'],
                     binding_sha256=identifier(binding),session=session,parent=parent,environment=environment)
        source=output/name
        pending.rename(source)
        pin=relay.seal(source,details)
        relay.copy(source,pin,backup/name)
        published.append(dict(name=name,receipt_sha256=pin,step=step,kind=kind))
        print(f'{name}: {pin}',flush=True)
        pending=output/'pending'
        pending.mkdir()

    def evaluate(step):
        for name,reader,entries in [('fresh_val25',data,schedule['val']),
                                    ('historical_val18',historical,schedule['historical_val'])]:
            dest=pending/name
            dest.mkdir()
            def panel(item,*arrays):
                d5.save_panel(dest/f"{item['image_id']}.png",*arrays)
                resource()
            result=d5.evaluate(model,reader,entries,entries,device,panel if step==limit else None)
            result.pop('caveats')  # D5's hardcoded caveats belong to historical Val18 only.
            result['cohort']=name
            json_new(dest/'metrics.json',result)
            resource()

    def commit(step):
        resource()
        state=snapshot(model,optimizer,device,binding,schedule,session,environment,parent,step)
        _checkpoint(pending/'checkpoint.pth',state)
        json_new(pending/'boundary.json',dict(completed=step,next_item=next_item(schedule,step),
                 elapsed_seconds=clock()-started,memory=resource(),binding=binding))
        push('checkpoint',step)

    stop='SEGMENT_UPDATE_LIMIT'
    try:
        resource()
        if clock()>=deadline:
            stop='TIME_RESERVE'
        elif not parent:
            json_new(pending/'schedule.json',schedule)
            json_new(pending/'binding.json',binding)
            evaluate(0)
            push('evaluation',0)
        target=min(limit,completed+updates)
        for item in schedule['train'][completed:target]:
            if clock()>=deadline:
                stop='TIME_RESERVE'
                break
            resource()
            x,y,_=data.patch(item,'train')
            if clock()>=deadline:
                stop='TIME_RESERVE'
                break
            before=clock()
            loss=d5.update(model,optimizer,x.to(device),y.to(device))
            completed=item['step']
            json_new(pending/f'update-{completed:04d}.json',dict(item,loss=loss,seconds=clock()-before,memory=resource()))
            if completed==limit or completed%binding['config']['eval_every']==0:
                evaluate(completed)
            if completed==limit or completed%binding['config']['save_every']==0:
                commit(completed)
                last=completed
        if completed>last:
            commit(completed)
            last=completed
        if completed==limit:
            stop='RUN_COMPLETE'
    except BaseException as exc:
        json_new(output/'error.json',dict(type=type(exc).__name__,message=str(exc),
                                         last_published=last,published=published))
        raise
    result=dict(status=stop,completed=completed,session=session,parent=parent,published=published,
                wall_seconds=clock()-started,memory=resource())
    json_new(pending/'result.json',result)
    push('segment',completed)
    return result


def compare_cohorts(initial, initial_sha, final, final_sha):
    a,b=relay.verify(initial,initial_sha),relay.verify(final,final_sha)
    before,after=a['details'],b['details']
    need(before['binding_sha256']==after['binding_sha256'] and before['run_id']==after['run_id']
         and before['kind']=='evaluation' and before['completed']==0 and after['kind']=='checkpoint',
         'Comparison lineage mismatch')
    binding=strict_json((final/'boundary.json').read_bytes())['binding']
    need(identifier(binding)==after['binding_sha256'] and binding['config']['total']==after['completed'],
         'Final training checkpoint required for comparison')
    cohorts={}
    for name in ('fresh_val25','historical_val18'):
        av,bv=[strict_json((p/name/'metrics.json').read_bytes())['windows'] for p in (initial,final)]
        keys=('image_id','oam_id','crop_xywh')
        need([[r[k] for k in keys] for r in av]==[[r[k] for k in keys] for r in bv],
             'Comparison crop binding mismatch')
        result=comparison(av,bv)
        cohorts[name]=dict(step0=result['model_a']['aggregate'],final=result['model_b']['aggregate'],
                           final_minus_step0=result['delta_b_minus_a'])
    return dict(run_id=after['run_id'],final_step=after['completed'],cohorts=cohorts,
                caveats=['Exploratory same-cohort differences; no performance gate.',
                         'Dataset and update count differ from D5; no controlled isolation of data effect.'])


def runtime():
    import importlib.metadata as meta
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA device required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    version=meta.version('satlaspretrain-models')
    need(version=='0.3.1','SatlasPretrain 0.3.1 required')
    gpu=torch.cuda.get_device_properties(0)
    return dict(software=dict(python=platform.python_version(),cuda=torch.version.cuda,
                              cudnn=torch.backends.cudnn.version(),
                              packages={p:meta.version(p) for p in
                              ('torch','torchvision','satlaspretrain-models','numpy','Pillow','rasterio','pyproj')}),
                device=gpu.name,total_vram=gpu.total_memory)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--bundle-sha256',required=True,help='Expected unpacked bundle.json SHA')
    sub=ap.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('plan')
    p.add_argument('--run-id',required=True)
    p.add_argument('--total',type=int,default=700)
    p.add_argument('--save-every',type=int,default=175)
    p.add_argument('--eval-every',type=int,default=175)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('segment')
    for key in ('plan','initializer','output','backup','resume'):
        p.add_argument('--'+key,type=Path,required=key!='resume')
    for key in ('plan-sha256','session'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--receipt-sha256')
    for key in ('updates','max-vram'):
        p.add_argument('--'+key,type=int,required=True)
    for key in ('seconds','reserve'):
        p.add_argument('--'+key,type=float,required=True)
    args=ap.parse_args()
    need(sha(args.bundle/'bundle.json')==args.bundle_sha256,'Bundle control SHA mismatch')
    spec,data,val=read_bundle(args.bundle)
    need(git('rev-parse','HEAD')==spec['commit'] and not git('status','--porcelain'),'Pinned clean Git checkout required')
    if args.mode=='plan':
        binding,schedule=make_binding(spec,args.bundle_sha256,data,val,run_id=args.run_id,
                                      total=args.total,save_every=args.save_every,eval_every=args.eval_every)
        json_new(args.output,binding)
        print(f'Run plan SHA: {sha(args.output)}; schedule SHA: {identifier(schedule)}')
        return
    binding=pinned(args.plan,args.plan_sha256)
    cfg=binding['config']
    rebuilt,schedule=make_binding(spec,args.bundle_sha256,data,val,run_id=binding['run_id'],
                                  total=cfg['total'],save_every=cfg['save_every'],eval_every=cfg['eval_every'])
    need(rebuilt==binding,'Frozen run plan mismatch')
    need(sha(args.initializer)==binding['initializer_sha256'],'Initializer SHA mismatch')
    environment=runtime()

    def model():
        import satlaspretrain_models as satlas
        weights=verified_load(args.initializer,binding['initializer_sha256'],None,torch)
        return satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,
                            fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights)

    result=run_segment(data,val,binding,schedule,model,'cuda',environment,args.session,args.output,args.backup,
                updates=args.updates,seconds=args.seconds,reserve=args.reserve,max_vram=args.max_vram,
                resume=args.resume,receipt_sha=args.receipt_sha256,
                protected=(args.bundle,args.initializer,args.plan))
    print(canonical(result).decode(),end='')


if __name__=='__main__':
    main()
