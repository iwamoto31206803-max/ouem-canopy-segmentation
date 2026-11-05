"""Opt-in official pretrained CPU smoke on hash-verified real data."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import time

from m1_1_satlas_smoke import checkpoint_digest, validate_output, gradient_evidence
from m1_2d_geometry_audit import sha
from m1_3a_loader import load_patch


def run(root, checkpoint, record):
    import torch
    import satlaspretrain_models as satlas
    import satlaspretrain_models.models.heads as heads
    pairs = json.loads(Path('docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    provenance = json.loads(Path('docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    record.update(python=platform.python_version(), platform=platform.platform(),
                  versions={p:version(p) for p in ['torch','torchvision','satlaspretrain-models','rasterio','numpy']},
                  checkpoint=provenance, pair_manifest_sha256=sha(Path('docs/implementation/M1_2D_PAIR_RESULT.json')),
                  official_head_source_sha256=sha(Path(inspect.getfile(heads))))
    checkpoint_digest(checkpoint, provenance['sha256'])
    if len({s['oam_id'] for s in pairs['samples']}) != len(pairs['samples']):
        raise ValueError('Train/Val source-group overlap')
    # All validation occurs before constructing the heavy official architecture.
    patches = [(sample, load_patch(root, sample, sample['split'], 256 if sample['split']=='train' else 224))
               for sample in pairs['samples']]
    torch.set_num_threads(4)
    torch.manual_seed(0)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not weights or not all(isinstance(v,torch.Tensor) for v in weights.values()):
        raise ValueError('Unexpected checkpoint state dict')
    model = satlas.Model(num_channels=3, multi_image=False, backbone=satlas.Backbone.SWINB,
                         fpn=True, head=satlas.Head.SEGMENT, num_categories=2, weights=weights).cpu()
    del weights
    model.requires_grad_(True)
    record['components'] = dict(backbone='strict_loaded',fpn='strict_loaded',upsample='random',head='random')
    record['cases'] = []
    for sample, (images, targets, crop) in patches:
        start = time.perf_counter()
        model.train(sample['split']=='train')
        model.zero_grad(set_to_none=True)
        probabilities, loss = model(images, targets)
        validate_output(torch, probabilities, loss, images.shape[-1])
        loss.backward()
        record['cases'].append(dict(image_id=sample['image_id'],split=sample['split'],
            model_mode='train' if model.training else 'eval', crop=[0,0,images.shape[-1],images.shape[-2]],
            input_shape=list(images.shape),target_shape=list(targets.shape), output_shape=list(probabilities.shape),
            input_dtype=str(images.dtype), target_dtype=str(targets.dtype), input_range=[images.min().item(),images.max().item()],
            target_values=targets.unique().tolist(),loss=loss.item(),
            gradients={name:gradient_evidence(torch,getattr(model,name)) for name in ['backbone','fpn','head']},
            seconds=time.perf_counter()-start, source_hashes=sample['output_hashes'],padding=False))
        del probabilities, loss
    record['status'] = 'REAL_DATA_CPU_PASS'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    record=dict(status='NOT_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),device='cpu',batch=1,threads=4,seed=0)
    start=time.perf_counter()
    code=0
    try:
        run(args.data,args.checkpoint,record)
    except Exception as error:
        record.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        code=1
    record['total_seconds']=time.perf_counter()-start
    record['memory_measurement']='NOT_MEASURED'
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
