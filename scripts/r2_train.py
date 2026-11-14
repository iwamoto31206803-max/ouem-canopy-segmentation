"""R1a-backed CPU training core. The caller supplies a Satlas-style model factory."""
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

import r1_dataset as r1
from m1_6_colab_gpu_smoke import safe_load, save_state
from m1_7_g2_train import digest, write_new


KIND = 'OUEM_R2A1_CPU_CORE'


class R1Data:
    def __init__(self, root, manifest, dataset_id, manifest_sha256):
        manifest = Path(manifest)
        r1.require_hash(manifest, manifest_sha256)
        value = r1.read_manifest(manifest)
        if value.get('dataset_id') != dataset_id:
            raise ValueError('Dataset ID mismatch')
        self.verified = r1.preflight(Path(root), value)
        self.root = self.verified.root
        r1.require_hash(manifest, manifest_sha256)
        self.samples = {s['image_id']: s for s in json.loads(self.verified.manifest_bytes)['samples']}
        self.identity = dict(dataset_id=dataset_id, manifest_sha256=manifest_sha256,
                             source_generation=value['source_generation'])

    def patch(self, item, split):
        sample = self.samples.get(item['image_id'])
        if sample is None or sample['split'] != split or sample['oam_id'] != item['oam_id']:
            raise ValueError('Sample/source/split mismatch')
        x, y, width, height = item['crop_xywh']
        h, w = sample['grid']['shape']
        if (any(type(v) is not int for v in item['crop_xywh']) or min(x, y) < 0
                or min(width, height) <= 0 or x+width > w or y+height > h):
            raise ValueError('Crop outside image; padding/upscale forbidden')
        arrays = r1.load_sample(self.verified, item['image_id'], split)
        crops = {name: array[:, y:y+height, x:x+width].copy() for name, array in arrays.items()}
        if not np.all(crops['valid_fraction.tif'] == 1):
            raise ValueError('Patch validity HOLD')
        images = torch.from_numpy(crops['rgb.tif']).unsqueeze(0).to(torch.float32)/255
        target = torch.from_numpy(crops['canopy_mask.tif']).to(torch.long)
        return images, target, crops


def make_plan(data, *, seed=0, patch_hw=(224, 224), steps=20, val_ids=None,
              val_origin=(0, 0), lr=1e-4, weight_decay=.01):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(steps) is not int or steps <= 0:
        raise ValueError('Integer seed and positive step budget required')
    if (len(patch_hw) != 2 or any(type(v) is not int or v <= 0 for v in patch_hw)
            or len(val_origin) != 2 or any(type(v) is not int or v < 0 for v in val_origin)):
        raise ValueError('Positive integer patch and nonnegative Val origin required')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (lr, weight_decay)) or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid AdamW settings')
    train_ids = sorted(i for i, s in data.samples.items() if s['split'] == 'train')
    val_ids = sorted(i for i, s in data.samples.items() if s['split'] == 'val') if val_ids is None else list(val_ids)
    if not train_ids or len(set(val_ids)) != len(val_ids) or any(
            type(i) is not int or i not in data.samples or data.samples[i]['split'] != 'val' for i in val_ids):
        raise ValueError('Train samples and distinct Val-only IDs required')
    ph, pw = patch_hw
    positions = getattr(data, 'crop_positions', None)
    if positions is not None and (list(patch_hw) != [224,224] or list(val_origin) != [0,0]):
        raise ValueError('Research pool uses stored 224 windows and fixed Val positions')
    for rid in train_ids + val_ids:
        h, w = data.samples[rid]['grid']['shape']
        x, y = val_origin if rid in val_ids else (0, 0)
        if h < y+ph or w < x+pw:
            raise ValueError(f'Image {rid} smaller than requested patch/origin')
    def entry(rid, x, y):
        return dict(image_id=rid, oam_id=data.samples[rid]['oam_id'], crop_xywh=[x, y, pw, ph])
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < steps:
        cycle = list(train_ids)
        rng.shuffle(cycle)
        for rid in cycle[:steps-len(schedule)]:
            h, w = data.samples[rid]['grid']['shape']
            x, y = (positions[rid][rng.randrange(len(positions[rid]))] if positions is not None
                    else (rng.randint(0, w-pw), rng.randint(0, h-ph)))
            schedule.append(dict(step=len(schedule)+1, **entry(rid, x, y)))
    config = dict(seed=seed, patch_hw=list(patch_hw), steps=steps, val_ids=val_ids,
                  val_origin=list(val_origin), lr=lr, weight_decay=weight_decay)
    return dict(inputs=dict(data.identity), config=config, train=schedule,
                val=[entry(rid, *(positions[rid][0] if positions is not None else val_origin)) for rid in val_ids])


def _rng_state():
    state = np.random.get_state()
    return dict(cpu_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=[state[0], state[1].tolist(), *state[2:]])


def _restore_rng(state):
    torch.set_rng_state(state['cpu_rng'])
    random.setstate(state['python_rng'])
    n = state['numpy_rng']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


def _checkpoint(path, payload):
    partial = path.with_name(path.name+'.partial')
    sha = save_state(partial, payload)
    # Exclusive publication preserves a prior checkpoint and retains failed partials.
    os.link(partial, path)
    partial.unlink()
    return sha


def train(data, plan, model_factory, output, *, stop_after=None, resume=None, resume_sha256=None):
    """CPU FP32/AdamW with a caller model returning (probabilities, CE loss).

    Every segment uses a new output. Resume restores stored schedule and all CPU
    RNGs. The supplied model factory must preserve model semantics across resume.
    """
    if plan != make_plan(data, **plan['config']):
        raise ValueError('Plan seed/schedule/dataset mismatch')
    config = plan['config']
    end = config['steps'] if stop_after is None else stop_after
    if type(end) is not int or not 1 <= end <= config['steps']:
        raise ValueError('Stop step exceeds configured budget')
    state = None
    completed = 0
    if resume is not None:
        state = safe_load(Path(resume), resume_sha256)
        completed = state['completed_steps']
        if state['kind'] != KIND or state['plan'] != plan or type(completed) is not int or not 0 < completed < end:
            raise ValueError('Checkpoint kind/plan/step mismatch')
        if len(state['history']) != completed or any(
                h['step'] != i+1 or not math.isfinite(h['loss'])
                or any(h[k] != plan['train'][i][k] for k in ('image_id', 'oam_id', 'crop_xywh'))
                for i, h in enumerate(state['history'])):
            raise ValueError('Checkpoint history/schedule mismatch')
        if any(group['lr'] != config['lr'] or group['weight_decay'] != config['weight_decay']
               for group in state['optimizer']['param_groups']):
            raise ValueError('Checkpoint optimizer/config mismatch')
    output = Path(output).resolve()
    root = data.root
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Output overlaps dataset')
    if resume is not None and Path(resume).resolve().parent in [output.resolve(), *output.resolve().parents]:
        raise ValueError('Output overlaps checkpoint directory')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(kind=KIND, status='FAIL', plan=plan, completed_steps=completed,
                  history=list(state['history']) if state else [],
                  environment=dict(torch=str(torch.__version__), device='cpu', threads=torch.get_num_threads()))
    try:
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        model = model_factory()
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('CPU FP32 model required')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        if state:
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            _restore_rng(state)
        model.train()
        for number in range(completed+1, end+1):
            item = plan['train'][number-1]
            images, target, _ = data.patch(item, 'train')
            optimizer.zero_grad(set_to_none=True)
            probability, loss = model(images, target)
            if probability.shape != (1, 2, *config['patch_hw']) or loss.ndim != 0 or not torch.isfinite(loss) or not torch.isfinite(probability).all():
                raise ValueError('Invalid model output/loss')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError('Missing/nonfinite gradients')
            optimizer.step()
            values = list(model.parameters()) + [v for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)]
            if not all(torch.isfinite(v).all() for v in values):
                raise ValueError('Nonfinite model/optimizer')
            entry = dict(item, loss=float(loss.detach()), positive_fraction=float(target.float().mean()))
            record['history'].append(entry)
            record['completed_steps'] = number
            write_new(output/f'step_{number:06d}.json', entry)
        payload = dict(kind=KIND, plan=plan, completed_steps=end, history=record['history'],
                       model=model.state_dict(), optimizer=optimizer.state_dict(), **_rng_state())
        sha = _checkpoint(output/'checkpoint.pth', payload)
        record.update(status='CPU_TRAINING_CORE_PASS', checkpoint_sha256=sha,
                      schedule_sha256=digest(plan['train']), next_step=end+1)
    except Exception as exc:
        record['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        write_new(output/'result.json', record)
    return record
