"""Opt-in Zenodo10 FP32 CUDA preflight / ONE optimizer step / restore-only check."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import traceback

import numpy as np
import rasterio
import m1_7_dataset as data
from m1_6_colab_gpu_smoke import cuda_memory, timed_cuda, safe_load, save_state

REPO=data.REPO
KIND='OUEM_M1_7_G1B_ZENODO10_STEP1'
COMPONENTS=['backbone','fpn','upsample','head']
CONFIG=dict(seed=0,patch=224,batch_size=1,steps=1,precision='FP32',lr=1e-4,weight_decay=.01,
            betas=[.9,.999],eps=1e-8,threads=4,architecture='Aerial_SwinB_SI/official-FPN/SEGMENT-2',
            crop_origin=[0,0],scheduler=False,augmentation=False,padding=False)


def load_patch(root, manifest, image_id=3, role='train', crop_xywh=None):
    """Verification is deliberately full-dataset, even for a single selected crop."""
    import torch
    data.verify(root,manifest,data.MANIFEST_SHA256)
    m=json.loads(manifest.read_text());sample=next((s for s in m['samples'] if s['image_id']==image_id),None)
    if role not in ('train','val') or sample is None or sample['split']!=role: raise ValueError('Image/role mismatch')
    crops={};size=CONFIG['patch'];top,left=CONFIG['crop_origin']
    if crop_xywh is not None:
        if len(crop_xywh)!=4 or any(type(v) is not int for v in crop_xywh):raise ValueError('Integer crop required')
        left,top,width,height=crop_xywh
        if left<0 or top<0 or width!=size or height!=size:raise ValueError('Unsupported crop bounds/size')
    for name in sorted(data.PRODUCTS):
        with rasterio.open(root/role/str(image_id)/name) as src:
            if src.height<top+size or src.width<left+size: raise ValueError('Patch outside valid extent')
            crops[name]=src.read(window=rasterio.windows.Window(left,top,size,size))
    rgb=crops['rgb.tif'];mask=crops['canopy_mask.tif'];valid=crops['valid_fraction.tif']
    if rgb.shape!=(3,size,size) or rgb.dtype!=np.uint8 or mask.shape!=(1,size,size) or mask.dtype!=np.uint8:
        raise ValueError('Patch shape/dtype mismatch')
    if not np.isin(mask,[0,1]).all() or not np.all(valid==1): raise ValueError('Invalid target/validity')
    images=torch.from_numpy(rgb.copy()).unsqueeze(0).to(torch.float32)/255
    target=torch.from_numpy(mask.copy()).to(torch.long)
    if images.shape!=(1,3,size,size) or target.shape!=(1,size,size) or not torch.isfinite(images).all() or not ((images>=0)&(images<=1)).all():
        raise ValueError('Invalid tensors')
    inputs=dict(dataset_id=m['dataset_id'],source_generation=m['source_generation'],manifest_sha256=data.MANIFEST_SHA256,
        image_id=image_id,oam_id=sample['oam_id'],split=role,validation_fold=sample['validation_fold'],
        product_hashes=sample['output_hashes'],crop_xywh=[left,top,size,size],rgb_shape=list(images.shape),
        target_shape=list(target.shape),rgb_dtype=str(images.dtype),target_dtype=str(target.dtype),
        rgb_range=[images.min().item(),images.max().item()],positive_fraction=float(target.float().mean()),
        valid_fraction_range=[float(valid.min()),float(valid.max())])
    return images,target,crops,inputs


def g1a_receipt(path):
    r=json.loads(path.read_text());expected=json.loads((REPO/'docs/implementation/M1_7_G1A_TRANSFER.json').read_text())
    if r.get('status')!='PASS' or r.get('zip_sha256')!=expected['zip_sha256'] or r.get('zip_bytes')!=expected['zip_bytes'] or r.get('file_count')!=41:
        raise ValueError('G1-A transfer receipt not PASS for fixed ZIP')
    d=r.get('dataset',{})
    if any(d.get(k)!=v for k,v in dict(status='PASS',dataset_id=data.DATASET_ID,manifest_sha256=data.MANIFEST_SHA256,products=40,train=7,val=3).items()):
        raise ValueError('G1-A dataset receipt mismatch')
    return dict(sha256=data.sha(path),status='PASS',dataset_id=d['dataset_id'])


def copy_backup(checkpoint, expected_sha256, destination):
    data.require_hash(checkpoint,expected_sha256)
    data.fresh_external(destination,[checkpoint])
    target=destination/checkpoint.name
    with checkpoint.open('rb') as source,target.open('xb') as output: shutil.copyfileobj(source,output,1024**2)
    data.require_hash(target,expected_sha256);data.require_hash(checkpoint,expected_sha256)
    return dict(filename=target.name,sha256=expected_sha256,bytes=target.stat().st_size,copy_sha_match=True,
                destination_directory=str(destination.resolve()),storage='caller-supplied; mount/durability must be user-verified')


def fingerprints(model, optimizer, torch):
    from m1_4_bounded_overfit import fingerprint
    return dict(model=fingerprint(model.state_dict()),optimizer=fingerprint(optimizer.state_dict()),
                cpu_rng=fingerprint(torch.get_rng_state()),cuda_rng=fingerprint(torch.cuda.get_rng_state_all()),
                python_rng=fingerprint(random.getstate()))


def compatible(state,inputs,initializer,environment,git_sha,allow_change=False):
    if state.get('kind')!=KIND or state.get('config')!=CONFIG or type(state.get('completed_steps')) is not int or state['completed_steps']!=1:
        raise ValueError('Checkpoint kind/config/completed step mismatch')
    if state.get('inputs')!=inputs or state.get('initializer')!=initializer:
        raise ValueError('Checkpoint dataset/patch/initializer identity mismatch')
    changes={k:[state['environment'].get(k),v] for k,v in environment.items() if state['environment'].get(k)!=v}
    if state['git_sha']!=git_sha:changes['git_sha']=[state['git_sha'],git_sha]
    if changes and not allow_change:raise ValueError('Environment/commit changed; explicit --allow-environment-change required')
    return changes


def restore(model,optimizer,state,torch):
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    if len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device-count mismatch')
    torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
    observed=fingerprints(model,optimizer,torch)
    if observed!=state['fingerprints']:raise ValueError('Restored model/optimizer/RNG fingerprint mismatch')
    from m1_4_bounded_overfit import finite_parameters
    finite_parameters(model);finite_optimizer(optimizer,torch)
    return {k:'PASS' for k in observed}


def finite_optimizer(optimizer,torch):
    if not optimizer.state:raise ValueError('Missing optimizer state')
    for state in optimizer.state.values():
        if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Nonfinite optimizer state')


def one_step(model,optimizer,images,target,torch,record,evaluate_before=True):
    from m1_1_satlas_smoke import gradient_evidence,validate_output
    from m1_4_bounded_overfit import finite_parameters
    if evaluate_before:
        model.eval();record['stage']='initial_eval'
        with torch.no_grad():
            (probability,loss),seconds=timed_cuda(torch,lambda:model(images,target))
            record['nan_inf_detected']=not bool(torch.isfinite(probability).all() and torch.isfinite(loss))
            validate_output(torch,probability,loss,CONFIG['patch'])
            record['before_loss']=dict(value=loss.item(),mode='eval/no_grad',seconds=seconds)
        del probability,loss
    model.train();optimizer.zero_grad(set_to_none=True)
    record['stage']='forward_loss'
    (probability,loss),record['forward_seconds']=timed_cuda(torch,lambda:model(images,target))
    record['nan_inf_detected']=not bool(torch.isfinite(probability).all() and torch.isfinite(loss))
    validate_output(torch,probability,loss,CONFIG['patch'])
    record.update(step_loss=loss.item(),step_loss_mode='train',output_shape=list(probability.shape),
                  memory_after_forward=cuda_memory(torch))
    record['stage']='backward'
    _,record['backward_seconds']=timed_cuda(torch,loss.backward)
    record['gradients']={c:gradient_evidence(torch,getattr(model,c)) for c in COMPONENTS}
    representatives={}
    for component,evidence in record['gradients'].items():
        parameter=dict(getattr(model,component).named_parameters())[evidence['example']]
        representatives[component]=(evidence['example'],parameter,parameter.detach().cpu().clone())
    record['memory_after_backward']=cuda_memory(torch)
    record['stage']='optimizer_step'
    _,record['optimizer_seconds']=timed_cuda(torch,optimizer.step)
    finite_parameters(model);finite_optimizer(optimizer,torch)
    record['updates']={c:dict(parameter=name,changed=not torch.equal(p.detach().cpu(),before),
                             max_abs_change=float((p.detach().cpu()-before).abs().max())) for c,(name,p,before) in representatives.items()}
    if not all(item['changed'] for item in record['updates'].values()):raise ValueError('Active component representative not updated')
    record.update(completed_steps=1,optimizer_state_entries=len(optimizer.state),memory_after_step=cuda_memory(torch),
                  optimizer_step_status='ZENODO_GPU_OPTIMIZER_STEP_PASS',nan_inf_detected=False)


def run(args,record):
    import torch
    record['stage']='environment'
    record['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),cuda_build=torch.version.cuda,
                               platform=platform.platform(),gpu='NOT_MEASURED',total_vram_bytes='NOT_MEASURED',
                               **{'satlaspretrain-models':'NOT_MEASURED','torchvision':'NOT_MEASURED'})
    record['cuda_available']=torch.cuda.is_available()
    record['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if not record['cuda_available']:
        record['status']='CUDA_NOT_VERIFIED';raise RuntimeError('CUDA unavailable; no CPU training fallback')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Use a clean reviewed commit')
    torch.cuda.set_device(0);properties=torch.cuda.get_device_properties(0)
    record['environment'].update(gpu=properties.name,total_vram_bytes=properties.total_memory,device_count=torch.cuda.device_count())
    record['memory_before_model']=cuda_memory(torch)
    record['stage']='imports'
    import satlaspretrain_models as satlas
    import satlaspretrain_models.model as official_model
    import satlaspretrain_models.models.heads as official_heads
    record['environment'].update({p:version(p) for p in ['torchvision','satlaspretrain-models','numpy','rasterio']})
    if record['environment']['satlaspretrain-models']!='0.3.1':raise ValueError('Requires satlaspretrain-models 0.3.1')
    record['official_source_sha256']={n:data.sha(Path(inspect.getfile(m))) for n,m in [('model',official_model),('head',official_heads)]}
    record['environment']['official_source_sha256']=record['official_source_sha256']
    record['stage']='data_gate'
    record['g1a_transfer_verification']=g1a_receipt(args.g1a_result)
    images,target,_,inputs=load_patch(args.data,args.data/'manifest.json',args.image_id,'train')
    from m1_4_bounded_overfit import fingerprint
    inputs['tensor_fingerprint']=fingerprint({'images':images,'target':target})
    record['inputs']=inputs
    record['stage']='initializer_gate'
    initializer=json.loads((REPO/'docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    from m1_1_satlas_smoke import checkpoint_digest
    checkpoint_digest(args.checkpoint,initializer['sha256']);record['initializer']=initializer
    torch.set_num_threads(CONFIG['threads']);torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    record['environment']['runtime_flags']=dict(tf32=False,amp=False,cudnn_benchmark=False,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cpu_threads=torch.get_num_threads(),cudnn_version=torch.backends.cudnn.version())
    record['gates']=dict(environment='PASS',data40='PASS',initializer_hash='PASS')
    if args.preflight_only:
        record.update(status='PREFLIGHT_PASS_GPU_STEP_NOT_RUN')
        return
    torch.cuda.reset_peak_memory_stats();record['stage']='model_construction'
    state=None
    if args.restore:
        state=safe_load(args.restore,args.restore_sha256)
        record['environment_changes']=compatible(state,inputs,initializer,record['environment'],record['git_sha'],args.allow_environment_change)
        weights=None
    else:
        weights=safe_load(args.checkpoint,initializer['sha256'])
        if not isinstance(weights,dict) or not weights or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in weights.items()):
            raise ValueError('Official initializer must be a flat tensor dictionary')
    started=time.perf_counter()
    model=satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights)
    del weights
    model=model.to(device='cuda',dtype=torch.float32);model.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],betas=tuple(CONFIG['betas']),eps=CONFIG['eps'])
    torch.cuda.synchronize();record['construction_seconds']=time.perf_counter()-started
    record['memory_after_construction']=cuda_memory(torch)
    record['initialization']='restoring trained state' if state else 'official strict backbone/FPN; fresh Upsample/SEGMENT-2'
    if state:
        record['stage']='restore'
        record['restore_checks'],record['restore_seconds']=timed_cuda(torch,lambda:restore(model,optimizer,state,torch))
        record.update(status='ZENODO_GPU_CHECKPOINT_STATE_RESTORE_PASS',completed_steps=1,
                      restored_checkpoint_sha256=args.restore_sha256,continuation_determinism='NOT_VERIFIED: state restore only, no optimizer step')
        record['memory_after_restore']=cuda_memory(torch)
        return
    images=images.to('cuda');target=target.to('cuda')
    one_step(model,optimizer,images,target,torch,record)
    record['stage']='checkpoint_save'
    payload=dict(kind=KIND,config=CONFIG,completed_steps=1,inputs=inputs,initializer=initializer,
        model=model.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        python_rng=random.getstate(),git_sha=record['git_sha'],environment=record['environment'],
        fingerprints=fingerprints(model,optimizer,torch),g1a_transfer_verification=record['g1a_transfer_verification'])
    checkpoint=args.output/'zenodo10_step1.pth'
    digest,record['save_seconds']=timed_cuda(torch,lambda:save_state(checkpoint,payload))
    record['saved_checkpoint']=dict(filename=checkpoint.name,sha256=digest,bytes=checkpoint.stat().st_size,fingerprints=payload['fingerprints'])
    record['memory_after_save']=cuda_memory(torch)
    record['stage']='checkpoint_backup'
    started=time.perf_counter()
    record['backup_checkpoint']=copy_backup(checkpoint,digest,args.backup)
    record['backup_seconds']=time.perf_counter()-started
    record.update(status='ZENODO_GPU_STEP1_PASS_RESTORE_PENDING',memory_final=cuda_memory(torch))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','checkpoint','g1a-result','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image-id',type=int,choices=sorted(i for i,r in data.ROLES.items() if r=='train'),default=3)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--preflight-only',action='store_true');modes.add_argument('--restore',type=Path)
    p.add_argument('--restore-sha256');p.add_argument('--backup',type=Path,help='Required for step: fresh persistent/Drive directory')
    p.add_argument('--allow-environment-change',action='store_true')
    return p


def reserve(args):
    if bool(args.restore)!=bool(args.restore_sha256):raise ValueError('--restore and --restore-sha256 required together')
    if args.allow_environment_change and not args.restore:raise ValueError('Environment override is restore-only')
    if args.backup and (args.preflight_only or args.restore):raise ValueError('Backup is step-only')
    if not args.preflight_only and not args.restore and not args.backup:raise ValueError('Step requires a fresh --backup directory')
    protected=[args.data,args.checkpoint,args.g1a_result]+([args.restore] if args.restore else [])
    if args.backup:
        # Check without creating Drive outputs until a local checkpoint has been saved.
        b=args.backup.resolve()
        if args.backup.exists() or args.backup.is_symlink() or b==REPO or REPO in b.parents:raise ValueError('Fresh external backup required')
        if any(b==x.resolve() or x.resolve() in b.parents or b in x.resolve().parents for x in protected+[args.output]):raise ValueError('Backup overlaps protected inputs/output')
        protected.append(args.backup)
    data.fresh_external(args.output,protected)


def main(argv=None):
    args=parser().parse_args(argv)
    try:reserve(args)
    except (ValueError,OSError) as exc:parser().error(str(exc))
    record=dict(status='NOT_VERIFIED',stage='startup',utc=datetime.now(timezone.utc).isoformat(),config=CONFIG,
        completed_steps=0,forward_seconds='NOT_MEASURED',backward_seconds='NOT_MEASURED',optimizer_seconds='NOT_MEASURED',
        nan_inf_detected='NOT_MEASURED',memory_before_model='NOT_MEASURED',restore_checks='NOT_MEASURED',
        before_loss='NOT_MEASURED',step_loss='NOT_MEASURED',gradients='NOT_MEASURED',updates='NOT_MEASURED',
        saved_checkpoint='NOT_MEASURED',backup_checkpoint='NOT_MEASURED',restore_seconds='NOT_MEASURED')
    started=time.perf_counter();code=0
    try:run(args,record)
    except Exception as exc:
        record['exception']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if 'nonfinite' in str(exc).lower():record['nan_inf_detected']=True
        if record['status']!='CUDA_NOT_VERIFIED':record['status']='FAIL_STOPPED'
        try:
            import torch
            if torch.cuda.is_available():record['memory_at_failure']=cuda_memory(torch)
        except Exception:pass
        code=2 if record['status']=='CUDA_NOT_VERIFIED' else 1
    record['total_seconds']=time.perf_counter()-started
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2));return code


if __name__=='__main__':raise SystemExit(main())
