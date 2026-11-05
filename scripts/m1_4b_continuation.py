"""Opt-in same-environment strict CPU continuation verification."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import torch
from m1_1_satlas_smoke import checkpoint_digest, gradient_evidence, validate_output
from m1_3a_loader import load_patch
from m1_4_bounded_overfit import CONFIG, COMPONENTS, construct, evaluate, finite_parameters, fingerprint, memory_guard
from m1_2d_geometry_audit import sha


def validate_metadata(state,baseline):
    if state['completed_steps']!=20: raise ValueError('Completed step mismatch')
    if state['config']!=CONFIG or state['seed']!=0: raise ValueError('Config/seed mismatch')
    for key in ['inputs','dataset_revision','initial_checkpoint','experiment']:
        if state[key]!=baseline[key]: raise ValueError('Provenance mismatch: '+key)


def state_fingerprints(model,optimizer):
    return dict(model=fingerprint(model.state_dict()),optimizer=fingerprint(optimizer.state_dict()),
                rng=fingerprint(torch.get_rng_state()))


def require_match(actual,expected):
    comparison={key:actual[key]==expected[key] for key in expected}
    if not all(comparison.values()):
        raise ValueError('Strict mismatch: '+','.join(k for k,v in comparison.items() if not v))
    return comparison


def step(model,optimizer,patch,inspect_update=False):
    model.train(); optimizer.zero_grad(set_to_none=True)
    images,target,_=patch
    before={n:fingerprint(p) for n,p in model.named_parameters()} if inspect_update else {}
    memory_before=memory_guard(); started=time.perf_counter()
    probability,loss=model(images,target)
    validate_output(torch,probability,loss,256)
    loss.backward()
    gradients={c:gradient_evidence(torch,getattr(model,c)) for c in COMPONENTS}
    gradient_hash=fingerprint({n:p.grad for n,p in model.named_parameters()}) if inspect_update else None
    value=loss.item()
    optimizer.step()
    finite_parameters(model)
    for state in optimizer.state.values():
        if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Nonfinite optimizer state')
    updated=[n for n,p in model.named_parameters() if before.get(n)!=fingerprint(p)] if inspect_update else []
    if inspect_update and not updated: raise ValueError('No actual parameter updates')
    return dict(loss=value,gradients=gradients,gradient_fingerprint=gradient_hash,
                after=state_fingerprints(model,optimizer) if inspect_update else None,
                updated_parameter_count=len(updated),updated_parameter_names_fingerprint=fingerprint(updated),
                seconds=time.perf_counter()-started,memory_before=memory_before,memory_after=memory_guard())


def run(args,record):
    baseline=json.loads(Path('docs/implementation/M1_4_TRAINING_RESULT.json').read_text())
    checkpoint_digest(args.trained,baseline['saved_checkpoint']['sha256'])
    checkpoint_digest(args.initial,baseline['initial_checkpoint']['sha256'])
    versions={p:version(p) for p in ['torch','torchvision','satlaspretrain-models']}
    if versions!=baseline['versions'] or platform.python_version()!=baseline['python']:
        raise ValueError('Environment version mismatch')
    pairs=json.loads(Path('docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    inputs=[dict(image_id=s['image_id'],oam_id=s['oam_id'],split=s['split'],hashes=s['output_hashes'],
                 license=s['source_license'],attribution=s['attribution_reference']) for s in pairs['samples']]
    if inputs!=baseline['inputs'] or len({s['oam_id'] for s in pairs['samples']})!=2:
        raise ValueError('Input provenance/split mismatch')
    patches={s['split']:load_patch(args.data,s,s['split'],256 if s['split']=='train' else 224) for s in pairs['samples']}
    torch.manual_seed(0); torch.set_num_threads(4)
    record.update(versions=versions,python=platform.python_version(),config=CONFIG,
                  provenance='PROVENANCE_VERIFIED',dataset_revision=pairs['revision'],
                  inputs=inputs,trained_sha256=baseline['saved_checkpoint']['sha256'],
                  initial_checkpoint=baseline['initial_checkpoint'],platform=platform.platform(),
                  runtime_flags=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                     mkldnn_enabled=torch.backends.mkldnn.enabled,threads=torch.get_num_threads()))
    expected={k:baseline['saved_checkpoint'][k+'_fingerprint'] for k in ['model','optimizer','rng']}
    if args.series=='reference':
        weights=torch.load(args.initial,map_location='cpu',weights_only=True)
        model,optimizer=construct(weights); del weights
        record['initial_eval']={split:evaluate(model,p) for split,p in patches.items()}
        record['reference_steps']=[]
        for number in range(1,21):
            entry=step(model,optimizer,patches['train'])
            record['reference_steps'].append(dict(step=number,loss=entry['loss'],seconds=entry['seconds'],memory=entry['memory_after']))
            print(json.dumps(dict(step=number,loss=entry['loss'])),flush=True)
        record['step20_final_eval']={split:evaluate(model,p) for split,p in patches.items()}
    else:
        state=torch.load(args.trained,map_location='cpu',weights_only=True)
        validate_metadata(state,baseline)
        model,optimizer=construct()
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng_state'])
        del state
    record['step20_state']=state_fingerprints(model,optimizer)
    record['step20_comparison']=require_match(record['step20_state'],expected)
    record['step20_status']='STEP20_BASELINE_MATCH'
    record['step21']=step(model,optimizer,patches['train'],True)
    record['status']='STEP21_MEASURED'
    if args.series=='resumed':
        reference=json.loads(args.reference.read_text())
        if reference['status']!='STEP21_MEASURED' or reference['trained_sha256']!=record['trained_sha256']:
            raise ValueError('Reference series provenance/status invalid')
        actual=record['step21']; expected_step=reference['step21']
        comparisons={k:actual[k]==expected_step[k] for k in ['loss','gradient_fingerprint','after',
                      'updated_parameter_count','updated_parameter_names_fingerprint']}
        record['step21_comparison']=comparisons
        if not all(comparisons.values()): raise ValueError('Step21 strict mismatch')
        record['status']='STEP21_CONTINUATION_MATCH'
    checkpoint_digest(args.trained,baseline['saved_checkpoint']['sha256'])
    record['original_checkpoint_preserved']=True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',choices=['reference','resumed'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--trained',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve()==repo or repo in args.output.resolve().parents:
        parser.error('Use fresh external output directory')
    if args.series=='resumed' and args.reference is None: parser.error('Reference JSON required')
    args.output.mkdir(parents=True)
    record=dict(series=args.series,status='NOT_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),gpu_vram='NOT_MEASURED')
    started=time.perf_counter(); code=0
    try: run(args,record)
    except Exception as error:
        record.update(status='FAIL_STOPPED',error=f'{type(error).__name__}: {error}'); code=1
    record['total_seconds']=time.perf_counter()-started
    try:
        record['final_memory']=memory_guard()
    except Exception as error:
        record.update(status='FAIL_STOPPED',memory_error=f'{type(error).__name__}: {error}')
        code=1
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    raise SystemExit(code)
