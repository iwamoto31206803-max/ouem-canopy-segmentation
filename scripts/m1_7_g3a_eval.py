"""Fixed Val3 eval-only comparison: reconstructed step0 versus pinned G2 step20."""
import argparse
from datetime import datetime, timezone
import gc
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
from PIL import Image,ImageDraw

import m1_7_dataset as data
import m1_7_g1b_gpu_smoke as shared
import m1_7_g2_train as g2

VAL_IDS=[1972,269,3785]
TRAINED_SHA256='47044e414c27718d37ea2d67a89f88cc24e2777757db07cfad2286ead2b9e767'
INITIAL_SHA256='f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036'
RATES=['tree_iou','f1','dice','precision','recall','truth_canopy_fraction','predicted_canopy_fraction','area_error','absolute_area_error']
COUNTS=['tp','fp','fn','tn','valid_pixels']
POLICY=dict(crop_xywh=[0,0,224,224],precision='FP32; no AMP/TF32',validity='all pixels including background0; valid_fraction==1',
    prediction='argmax(raw logits, dim=1); ties -> background0',probability='softmax(raw logits) class1',
    ce='raw-logit CE over all valid pixels; checked against official loss',undefined='null with zero-denominator reason',
    macro='mean of defined per-image rates; report contributing image count, exclude null',
    micro='sum TP/FP/FN/TN then recompute rates',aggregate_ce='sum(image CE * valid_pixels) / sum(valid_pixels)',
    deltas='B minus A; all fractional rates/errors in percentage points, CE in loss units, counts in pixels')


def count_metrics(tp,fp,fn,tn):
    n=tp+fp+fn+tn
    if any(type(v) is not int or v<0 for v in [tp,fp,fn,tn]) or n==0:raise ValueError('Nonnegative counts/nonempty evaluation required')
    formulas={'tree_iou':(tp,tp+fp+fn,'TP+FP+FN=0'),'f1':(2*tp,2*tp+fp+fn,'2TP+FP+FN=0'),
              'dice':(2*tp,2*tp+fp+fn,'2TP+FP+FN=0'),'precision':(tp,tp+fp,'TP+FP=0'),'recall':(tp,tp+fn,'TP+FN=0')}
    out=dict(tp=tp,fp=fp,fn=fn,tn=tn,valid_pixels=n,undefined_reasons={})
    for key,(a,b,reason) in formulas.items():
        out[key]=a/b if b else None
        if not b:out['undefined_reasons'][key]=reason
    out.update(truth_canopy_fraction=(tp+fn)/n,predicted_canopy_fraction=(tp+fp)/n,
               area_error=(fp-fn)/n,absolute_area_error=abs(fp-fn)/n)
    return out


def confusion(prediction,target):
    if prediction.shape!=target.shape or prediction.ndim!=2 or not np.isin(prediction,[0,1]).all() or not np.isin(target,[0,1]).all():
        raise ValueError('Aligned binary 2D arrays required')
    p=prediction==1;t=target==1
    return count_metrics(int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum()),int((~p&~t).sum()))


def aggregate(rows):
    if not rows:raise ValueError('No evaluated images')
    macro={};counts={}
    for key in RATES:
        values=[r[key] for r in rows if r[key] is not None];counts[key]=len(values)
        macro[key]=sum(values)/len(values) if values else None
    micro=count_metrics(*(sum(r[k] for r in rows) for k in ['tp','fp','fn','tn']))
    n=sum(r['valid_pixels'] for r in rows)
    return dict(macro=dict(metrics=macro,valid_image_counts=counts,total_images=len(rows),
                           undefined_reasons={k:'all image values undefined' for k,v in macro.items() if v is None}),
                micro=micro,ce_valid_pixel_mean=sum(r['ce']*r['valid_pixels'] for r in rows)/n,
                ce_valid_pixels=n,ce_policy=POLICY['aggregate_ce'])


def delta(a,b,keys=RATES):
    values={k:100*(b[k]-a[k]) if a[k] is not None and b[k] is not None else None for k in keys}
    return dict(percentage_points=values,undefined_reasons={k:'A or B undefined' for k,v in values.items() if v is None})


def comparison(a,b):
    if [r['image_id'] for r in a]!=[r['image_id'] for r in b]:raise ValueError('Comparison scene mismatch')
    per=[]
    for x,y in zip(a,b):
        if x['valid_pixels']!=y['valid_pixels'] or x['truth_canopy_fraction']!=y['truth_canopy_fraction']:raise ValueError('Comparison target/pixel mismatch')
        per.append(dict(image_id=x['image_id'],**delta(x,y),ce=y['ce']-x['ce'],
                        count_difference={k:y[k]-x[k] for k in COUNTS}))
    aa=aggregate(a);bb=aggregate(b)
    return dict(model_a=dict(label='RECONSTRUCTED_STEP0',per_image=a,aggregate=aa),model_b=dict(label='G2_STEP20',per_image=b,aggregate=bb),
        delta_b_minus_a=dict(per_image=per,macro=delta(aa['macro']['metrics'],bb['macro']['metrics']),
                            micro=dict(**delta(aa['micro'],bb['micro']),count_difference={k:bb['micro'][k]-aa['micro'][k] for k in COUNTS}),
                            ce_valid_pixel_mean=bb['ce_valid_pixel_mean']-aa['ce_valid_pixel_mean']),policy=POLICY)


def load_val(root):
    data.verify(root,root/'manifest.json',data.MANIFEST_SHA256)
    m=json.loads((root/'manifest.json').read_text());patches={}
    for rid in VAL_IDS:
        x,y,crops,meta=shared.load_patch(root,root/'manifest.json',rid,'val',[0,0,224,224])
        s=next(s for s in m['samples'] if s['image_id']==rid)
        meta.update(full_shape=s['output_shape'],evaluated_pixels=224**2,
                    excluded_pixels=s['output_shape'][0]*s['output_shape'][1]-224**2)
        patches[rid]=(x,y,crops,meta)
    return m,patches


def validate_checkpoint(state,manifest,initializer):
    if state.get('completed_steps')!=20:raise ValueError('Accepted step20 required')
    # Evaluation is intentionally a new commit/runtime, not a training resume.
    g2.compatible(state,manifest,g2.config(0),initializer,state['environment'],state['git_sha'])
    if state['environment'].get('satlaspretrain-models')!='0.3.1':raise ValueError('Training Satlas version mismatch')
    if not state.get('model') or not state.get('fingerprints',{}).get('model'):raise ValueError('Missing trained model/fingerprint')
    from m1_4_bounded_overfit import fingerprint
    if fingerprint(state['model'])!=state['fingerprints']['model']:raise ValueError('Checkpoint model fingerprint mismatch')
    return dict(kind=state['kind'],completed_steps=state['completed_steps'],training_git_sha=state['git_sha'],
                config=state['config'],training_environment=state['environment'],model_fingerprint=state['fingerprints']['model'])


def reconstructed(initializer_path,device):
    import torch
    import satlaspretrain_models as satlas
    from satlaspretrain_models.model import adjust_state_dict_prefix
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    # G2 reseeds after its private sampler/preflight, immediately before loading/constructing.
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);random.seed(0)
    weights=shared.safe_load(initializer_path,INITIAL_SHA256)
    if not isinstance(weights,dict) or not weights or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in weights.items()):
        raise ValueError('Official flat tensor initializer required')
    model=satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights)
    checks={}
    for name,prefix,count in [('backbone','backbone.',1),('fpn','intermediates.0.',0)]:
        expected=adjust_state_dict_prefix(weights,name,prefix,count);actual=getattr(model,name).state_dict()
        if set(expected)!=set(actual) or any(not torch.equal(actual[k],expected[k]) for k in actual):raise ValueError('Official component weights differ: '+name)
        checks[name]=dict(status='EXACT_OFFICIAL_TENSORS',tensors=len(actual))
    return model.to(device=device,dtype=torch.float32),checks


def evaluate_model(model,patches,device='cuda'):
    import torch
    from m1_1_satlas_smoke import validate_output
    from m1_4_bounded_overfit import fingerprint,finite_parameters
    if list(patches)!=VAL_IDS:raise ValueError('Only fixed ordered Val3 may be evaluated')
    model.eval();model.requires_grad_(False);finite_parameters(model)
    before=fingerprint(model.state_dict());rows=[];arrays={};captured=[]
    def capture(module,args,output):captured.append(output)
    hook=model.head.layers.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for rid,(x,y,_,meta) in patches.items():
                if meta['split']!='val' or meta['crop_xywh']!=[0,0,224,224]:raise ValueError('Fixed Val crop required')
                captured.clear();x=x.to(device);y=y.to(device)
                probability,official_loss=model(x,y)
                if len(captured)!=1:raise ValueError('Expected one official head logits capture')
                logits=captured.pop()
                if logits.shape!=(1,2,224,224) or y.shape!=(1,224,224) or logits.dtype!=torch.float32 or not torch.isfinite(logits).all():
                    raise ValueError('Logit/target shape/dtype mismatch; resizing forbidden')
                p=logits.softmax(1);ce=torch.nn.functional.cross_entropy(logits,y,reduction='mean')
                validate_output(torch,probability,official_loss,224)
                if not torch.allclose(probability,p,rtol=1e-5,atol=1e-6) or not torch.allclose(official_loss,ce,rtol=1e-5,atol=1e-7):
                    raise ValueError('Official probabilities/CE differ from raw-logit calculation')
                pred=logits.argmax(1)[0].cpu().numpy().astype(np.uint8);truth=y[0].cpu().numpy()
                row=confusion(pred,truth);row.update(image_id=rid,ce=ce.item(),crop_xywh=meta['crop_xywh'])
                rows.append(row);arrays[rid]=dict(prediction=pred,probability=p[0,1].cpu().numpy().copy())
    finally:hook.remove()
    finite_parameters(model);after=fingerprint(model.state_dict())
    if before!=after:raise ValueError('Model state changed during evaluation')
    if model.training or any(m.training for m in model.modules()):raise ValueError('Evaluation mode changed')
    return rows,arrays,dict(before=before,after=after,unchanged=True,mode='eval+inference_mode; no optimizer')


def render(path,rgb,truth,a,b):
    """Native224 panels with a fixed shared palette; never alters evaluation inputs."""
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8 or truth.shape!=(224,224):raise ValueError('Native224 diagnostic arrays required')
    def binary(mask):return np.where(mask[:,:,None]==1,np.array([0,190,0],np.uint8),np.zeros(3,np.uint8)).astype(np.uint8)
    def errors(pred):
        p=pred==1;t=truth==1;out=np.full((*truth.shape,3),35,np.uint8)
        out[p&t]=[0,190,0];out[p&~t]=[230,0,180];out[~p&t]=[255,140,0];return out
    probability=np.repeat(np.rint(b['probability']*255).astype(np.uint8)[:,:,None],3,axis=2)
    panels=[('RGB',rgb),('GT: Tree/Canopy',binary(truth)),('A: reconstructed step0',binary(a['prediction'])),
            ('B: G2 step20',binary(b['prediction'])),('B: FP/FN error',errors(b['prediction'])),
            ('B: class1 probability',probability),('A: FP/FN error',errors(a['prediction']))]
    canvas=Image.new('RGB',(4*240,2*256),(245,245,245));draw=ImageDraw.Draw(canvas)
    for index,(title,array) in enumerate(panels):
        x=(index%4)*240+8;y=(index//4)*256+4;draw.text((x,y),title,fill='black');canvas.paste(Image.fromarray(array),(x,y+22))
    x=728;y=274
    for color,label in [((0,190,0),'Tree / TP'),((0,0,0),'Background mask'),((230,0,180),'FP: predicted tree'),((255,140,0),'FN: missed tree'),((35,35,35),'TN: true background')]:
        draw.rectangle((x,y,x+14,y+12),fill=color);draw.text((x+20,y),label,fill='black');y+=25
    draw.text((x,y+5),'Probability: 0 black / 1 white',fill='black')
    gradient=np.tile(np.arange(224,dtype=np.float32)/223,(16,1))
    canvas.paste(Image.fromarray(np.rint(gradient*255).astype(np.uint8)).convert('RGB'),(x,y+27))
    canvas.save(path)


def write_new(path,value):
    with path.open('xb') as stream:stream.write(data.encoded(value))


def publish(output,backup,result):
    names={'metrics.json','provenance.json'}|{f'val_{rid}.png' for rid in VAL_IDS}
    if {p.name for p in output.iterdir()}!=names or not all((output/n).is_file() for n in names):raise ValueError('Incomplete/extra evaluation payload')
    data.fresh_external(backup,[output]);result['backup_created']=True
    payload=[output/n for n in sorted(names)]
    result['artifacts']={};result['copy_checks']={}
    for source in payload:
        h=data.sha(source)
        with source.open('rb') as src,(backup/source.name).open('xb') as dst:shutil.copyfileobj(src,dst)
        data.require_hash(backup/source.name,h);data.require_hash(source,h)
        result['artifacts'][source.name]=h;result['copy_checks'][source.name]='SHA256_MATCH'
    result['status']='G3A_EVALUATION_AND_COPY_PASS'
    write_new(output/'result.json',result)
    h=data.sha(output/'result.json')
    with (output/'result.json').open('rb') as src,(backup/'result.json').open('xb') as dst:shutil.copyfileobj(src,dst)
    data.require_hash(backup/'result.json',h)
    hashes=dict(result['artifacts'],**{'result.json':h})
    receipt=dict(status='PASS',sha256=hashes,copy_checks={name:'SHA256_MATCH' for name in hashes},
                 self_hash_policy='sha256.json excluded from its own inventory; its local/backup bytes compared separately')
    write_new(output/'sha256.json',receipt)
    with (output/'sha256.json').open('rb') as src,(backup/'sha256.json').open('xb') as dst:shutil.copyfileobj(src,dst)
    data.require_hash(backup/'sha256.json',data.sha(output/'sha256.json'))


def run(args,result):
    started=time.perf_counter()
    import torch
    result['stage']='environment'
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),cuda_build=torch.version.cuda,
                     cuda_available=torch.cuda.is_available(),platform=platform.platform())
    result['environment']=environment
    if not environment['cuda_available']:raise RuntimeError('CUDA unavailable: GPU evaluation NOT_VERIFIED')
    torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
    environment.update(gpu=props.name,total_vram_bytes=props.total_memory)
    result['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=data.REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=data.REPO,text=True).strip():raise ValueError('Use a clean reviewed commit')
    import satlaspretrain_models as satlas
    import satlaspretrain_models.model as official_model
    import satlaspretrain_models.models.heads as official_heads
    environment.update({name:version(name) for name in ['satlaspretrain-models','torchvision','numpy','rasterio','Pillow']})
    if environment['satlaspretrain-models']!='0.3.1':raise ValueError('Satlas0.3.1 required')
    environment['official_source_sha256']={n:data.sha(Path(inspect.getfile(m))) for n,m in [('model',official_model),('head',official_heads)]}
    result['stage']='data_checkpoint_gates';manifest,patches=load_val(args.data)
    data.require_hash(args.initializer,INITIAL_SHA256)
    state=shared.safe_load(args.trained,TRAINED_SHA256)
    initializer=json.loads((data.REPO/'docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    training=validate_checkpoint(state,manifest,initializer)
    if training['training_environment'].get('official_source_sha256')!=environment['official_source_sha256']:
        raise ValueError('Official Satlas implementation differs from G2')
    provenance=dict(experiment_id=args.experiment_id,git_sha=result['git_sha'],environment=environment,
        dataset_id=data.DATASET_ID,source_generation=data.GENERATION,manifest_sha256=data.MANIFEST_SHA256,
        initializer_sha256=INITIAL_SHA256,g2_step20_sha256=TRAINED_SHA256,g2=training,
        model_a=dict(label='RECONSTRUCTED_STEP0',seed=0,initialization='G2 seed/order and official Backbone/FPN; fresh Upsample/Head',
                     exact_g2_step0_identity='NOT_VERIFIED: original step0 Head/Upsample snapshot absent; runtime/version/init-order limits'),
        val_ids=VAL_IDS,crops={str(i):p[3] for i,p in patches.items()},policies=POLICY,
        environment_difference={k:[training['training_environment'].get(k),v] for k,v in environment.items() if training['training_environment'].get(k)!=v},
        scope='three fixed Val crops only; no generalization/GSI accuracy claim')
    if args.preflight_only:
        write_new(args.output/'provenance.json',provenance);result['status']='G3A_PREFLIGHT_PASS_EVALUATION_NOT_RUN';return
    # Retain only model weights/identity; optimizer and RNG are never restored or advanced.
    weights_b=state['model'];fingerprint_b=state['fingerprints']['model'];del state
    result['stage']='model_a'
    model,checks=reconstructed(args.initializer,'cuda')
    a,arrays_a,check_a=evaluate_model(model,patches);del model;gc.collect();torch.cuda.empty_cache()
    provenance['model_a']['official_components']=checks;provenance['model_a']['evaluation_state']=check_a
    result['stage']='model_b'
    model=satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=None)
    model.load_state_dict(weights_b,strict=True);del weights_b
    from m1_4_bounded_overfit import fingerprint
    if fingerprint(model.state_dict())!=fingerprint_b:raise ValueError('Strict-loaded step20 fingerprint mismatch')
    model=model.to(device='cuda',dtype=torch.float32)
    b,arrays_b,check_b=evaluate_model(model,patches);del model;gc.collect();torch.cuda.empty_cache()
    if check_b['before']!=fingerprint_b:raise ValueError('Step20 state differs after device transfer')
    provenance['model_b']=dict(label='G2_STEP20',evaluation_state=check_b)
    result['stage']='metrics_render'
    write_new(args.output/'metrics.json',comparison(a,b));write_new(args.output/'provenance.json',provenance)
    for rid in VAL_IDS:
        rgb=patches[rid][2]['rgb.tif'].transpose(1,2,0);truth=patches[rid][1][0].numpy()
        render(args.output/f'val_{rid}.png',rgb,truth,arrays_a[rid],arrays_b[rid])
    result.update(stage='publish',gpu_evaluation='COMPLETED',evaluation_and_gates_seconds=time.perf_counter()-started)
    publish(args.output,args.backup,result)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','initializer','trained','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backup',type=Path);p.add_argument('--experiment-id',default='m1-7-g3a-01');p.add_argument('--preflight-only',action='store_true')
    return p


def reserve(args):
    protected=[args.data,args.initializer,args.trained]
    # A new child inside an old G2 lineage would still mutate that existing experiment.
    for destination in [args.output]+([args.backup] if args.backup else []):
        resolved=destination.resolve()
        if any((parent/'lineage.json').is_file() for parent in [resolved,*resolved.parents]):
            raise ValueError('Evaluation destinations must be outside existing G2 lineages')
    if not args.preflight_only and not args.backup:raise ValueError('Evaluation requires fresh backup destination')
    if args.backup:
        b=args.backup.resolve()
        if args.backup.exists() or args.backup.is_symlink() or b==data.REPO or data.REPO in b.parents:
            raise ValueError('Fresh external backup required')
        if any(b==s.resolve() or s.resolve() in b.parents or b in s.resolve().parents for s in protected+[args.output]):
            raise ValueError('Backup overlaps protected input/output')
        protected.append(args.backup)
    data.fresh_external(args.output,protected)


def main(argv=None):
    args=parser().parse_args(argv)
    try:reserve(args)
    except (ValueError,OSError) as exc:parser().error(str(exc))
    result=dict(status='NOT_VERIFIED',stage='startup',experiment_id=args.experiment_id,utc=datetime.now(timezone.utc).isoformat(),
                gpu_evaluation='NOT_VERIFIED',backup_created=False)
    started=time.perf_counter();code=0
    try:
        run(args,result)
    except (Exception,KeyboardInterrupt) as exc:
        result.update(status='FAIL_STOPPED',exception=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));code=1
        # Only this newly reserved run's report can be finalized/invalidated; old artifacts remain protected.
        (args.output/'result.json').write_bytes(data.encoded(result))
        if result['backup_created']:
            try:write_new(args.backup/'FAILURE.json',result)
            except Exception:pass
    result['total_seconds']=time.perf_counter()-started
    if args.preflight_only or code:
        (args.output/'result.json').write_bytes(data.encoded(result))
    print(json.dumps(dict(status=result['status'],result_json=str(args.output/'result.json'),error=result.get('exception',{}).get('message')),indent=2))
    return code


if __name__=='__main__':raise SystemExit(main())
