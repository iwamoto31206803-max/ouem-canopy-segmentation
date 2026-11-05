"""Explicit CPU-only, at most twenty-step real-data experiment and safe reload check."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import torch
import satlaspretrain_models as satlas
from m1_1_satlas_smoke import checkpoint_digest, gradient_evidence, validate_output
from m1_2d_geometry_audit import sha
from m1_3a_loader import load_patch
from m1_3b_resource_audit import memory

CONFIG = dict(seed=0, threads=4, batch_size=1, patch_size=256, lr=1e-4,
              weight_decay=0.01, betas=[0.9,0.999], eps=1e-8, steps=20, dtype='float32',device='cpu')
COMPONENTS = ['backbone','fpn','upsample','head']


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype),list(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item,(list,tuple)):
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def finite_parameters(model):
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameter')


def metrics(prediction,target):
    positive=prediction==1; truth=target==1
    tp=int((positive & truth).sum()); fp=int((positive & ~truth).sum()); fn=int((~positive & truth).sum())
    def ratio(a,b): return a/b if b else None
    return dict(tp=tp,fp=fp,fn=fn,iou=ratio(tp,tp+fp+fn),f1=ratio(2*tp,2*tp+fp+fn),
                precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn))


def evaluate(model,patch):
    images,target,_=patch
    model.eval()
    with torch.no_grad():
        probability,loss=model(images,target)
        validate_output(torch,probability,loss,images.shape[-1])
        return dict(ce=loss.item(),**metrics(probability.argmax(1),target))


def construct(weights=None, stages=None):
    started=time.perf_counter()
    model=satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,
                       fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights).cpu()
    model.requires_grad_(True)
    if stages is not None:
        stages['model_construction_seconds']=time.perf_counter()-started
        stages['model_constructed_memory']=memory_guard()
    started=time.perf_counter()
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],
                               betas=tuple(CONFIG['betas']),eps=CONFIG['eps'])
    if stages is not None:
        stages['optimizer_construction_seconds']=time.perf_counter()-started
        stages['optimizer_constructed_memory']=memory_guard()
    return model,optimizer


def memory_guard():
    observation=memory()
    if observation.get('committed_private_bytes',0)>8*1024**3:
        raise MemoryError('Experiment private-commit ceiling of 8 GiB exceeded')
    return observation


def train(args,record):
    if args.output.exists(): raise ValueError('Use fresh external experiment directory')
    repo=Path(__file__).resolve().parents[1]
    if args.output.resolve()==repo or repo in args.output.resolve().parents:
        raise ValueError('Checkpoint output must be outside repository')
    args.output.mkdir(parents=True)
    provenance=json.loads(Path('docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    checkpoint_digest(args.checkpoint,provenance['sha256'])
    pairs=json.loads(Path('docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    if len({s['oam_id'] for s in pairs['samples']})!=2: raise ValueError('Train/Val overlap')
    patches={s['split']:load_patch(args.data,s,s['split'],256 if s['split']=='train' else 224) for s in pairs['samples']}
    torch.manual_seed(0); torch.set_num_threads(4)
    record.update(config=CONFIG,initial_checkpoint=provenance,dataset_revision=pairs['revision'],
                  inputs=[dict(image_id=s['image_id'],oam_id=s['oam_id'],split=s['split'],hashes=s['output_hashes'],
                               license=s['source_license'],attribution=s['attribution_reference']) for s in pairs['samples']],
                  versions={p:version(p) for p in ['torch','torchvision','satlaspretrain-models']},python=platform.python_version())
    start=time.perf_counter()
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,optimizer=construct(weights,record); del weights
    record['construction_seconds']=time.perf_counter()-start
    record['model_optimizer_ready_memory']=memory_guard()
    record['initial_eval']={split:evaluate(model,patch) for split,patch in patches.items()}
    record['steps']=[]
    for step in range(1,21):
        model.train(); optimizer.zero_grad(set_to_none=True)
        images,target,_=patches['train']
        entry=dict(step=step,memory_before=memory_guard())
        start=time.perf_counter(); probability,loss=model(images,target)
        entry['forward_ce_seconds']=time.perf_counter()-start
        validate_output(torch,probability,loss,256)
        entry['loss']=loss.item(); entry['memory_after_forward']=memory_guard()
        start=time.perf_counter(); loss.backward()
        entry['backward_seconds']=time.perf_counter()-start
        entry['gradients']={c:gradient_evidence(torch,getattr(model,c)) for c in COMPONENTS}
        norm_squared=sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)
        entry['gradient_norm']=norm_squared**0.5
        if not torch.isfinite(torch.tensor(entry['gradient_norm'])): raise ValueError('Nonfinite gradient norm')
        representatives={}
        for c in COMPONENTS:
            name=entry['gradients'][c]['example']
            p=dict(getattr(model,c).named_parameters())[name]
            representatives[c]=(name,p,p.detach().clone())
        entry['memory_before_optimizer']=memory_guard()
        start=time.perf_counter(); optimizer.step()
        entry['optimizer_seconds']=time.perf_counter()-start
        finite_parameters(model)
        entry['memory_after_optimizer']=memory_guard()
        entry['updated']={c:dict(parameter=n,changed=not torch.equal(p.detach(),before),
                                max_abs_change=float((p.detach()-before).abs().max())) for c,(n,p,before) in representatives.items()}
        if not all(v['changed'] for v in entry['updated'].values()): raise ValueError('Active representative not updated')
        for state in optimizer.state.values():
            if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
                raise ValueError('Nonfinite optimizer state')
        if not optimizer.state: raise ValueError('Optimizer state missing')
        entry['optimizer_state_entries']=len(optimizer.state)
        record['steps'].append(entry)
        if step==1: record['optimizer_step_status']='OPTIMIZER_STEP_PASS'
        print(json.dumps(dict(step=step,loss=entry['loss'])),flush=True)
        del probability,loss,representatives
    record['final_eval']={split:evaluate(model,patch) for split,patch in patches.items()}
    record['overfit_diagnostic']='IMPROVED' if record['final_eval']['train']['ce']<record['initial_eval']['train']['ce'] else 'UNCHANGED_OR_WORSE'
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_steps=20,
               config=CONFIG,seed=0,torch_rng_state=torch.get_rng_state(),initial_checkpoint=provenance,
               dataset_revision=pairs['revision'],inputs=record['inputs'],experiment=record['experiment'])
    start=time.perf_counter()
    path=args.output/'m1_4_trained_step20.pth'
    torch.save(state,path)
    record['save_seconds']=time.perf_counter()-start
    record['memory_after_save']=memory_guard()
    record['saved_checkpoint']=dict(filename=path.name,sha256=sha(path),bytes=path.stat().st_size,
        model_fingerprint=fingerprint(state['model']),optimizer_fingerprint=fingerprint(state['optimizer']),
        rng_fingerprint=fingerprint(state['torch_rng_state']))
    record['status']='TRAINING_COMPLETE_RELOAD_PENDING'


def reload_check(args,record):
    torch.set_num_threads(4)
    baseline=json.loads((args.output/'training_result.json').read_text())
    checkpoint=args.output/baseline['saved_checkpoint']['filename']
    checkpoint_digest(checkpoint,baseline['saved_checkpoint']['sha256'])
    start=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (state['config']!=CONFIG or state['completed_steps']!=20 or state['inputs']!=baseline['inputs']
            or state['initial_checkpoint']!=baseline['initial_checkpoint'] or state['seed']!=CONFIG['seed']
            or state['dataset_revision']!=baseline['dataset_revision'] or state['experiment']!=baseline['experiment']):
        raise ValueError('Saved provenance/config mismatch')
    model,optimizer=construct()
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng_state'])
    checks=dict(model=fingerprint(model.state_dict())==baseline['saved_checkpoint']['model_fingerprint'],
                optimizer=fingerprint(optimizer.state_dict())==baseline['saved_checkpoint']['optimizer_fingerprint'],
                rng=fingerprint(torch.get_rng_state())==baseline['saved_checkpoint']['rng_fingerprint'])
    if not all(checks.values()): raise ValueError('Reload state mismatch')
    finite_parameters(model)
    record.update(status='CHECKPOINT_RESUME_PASS',checks=checks,completed_steps=state['completed_steps'],
                  checkpoint_sha256=sha(checkpoint),reload_seconds=time.perf_counter()-start,
                  memory_after_reload=memory_guard(),continuation_determinism='NOT_VERIFIED_LOAD_AND_STATE_COMPARISON_ONLY')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reload-check',action='store_true')
    args=parser.parse_args()
    record=dict(status='NOT_VERIFIED',experiment='M1-4-CPU-20steps-seed0',utc=datetime.now(timezone.utc).isoformat(),gpu_vram='NOT_MEASURED')
    if not args.reload_check and args.output.exists(): parser.error('Use fresh output to protect existing experiments')
    if not args.reload_check and (args.data is None or args.checkpoint is None): parser.error('Train requires data and checkpoint')
    code=0; start=time.perf_counter()
    try:
        if args.reload_check: reload_check(args,record)
        else: train(args,record)
    except Exception as error:
        record.update(status='FAIL_STOPPED',error=f'{type(error).__name__}: {error}'); code=1
    record['total_seconds']=time.perf_counter()-start
    name='reload_result.json' if args.reload_check else 'training_result.json'
    if args.output.exists(): (args.output/name).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    raise SystemExit(code)
