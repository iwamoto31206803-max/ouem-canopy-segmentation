"""Fixed v5 Satlas experiment.

Default mode checks the runtime and pinned inputs. --execute starts or resumes
training. --verify-restore checks a checkpoint in a separate process.
Schedule SHA uses G2's canonical JSON encoding. Time and reserved VRAM are
checked between operations. Resume keeps the original wall-clock deadline.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from m1_4_bounded_overfit import finite_parameters, fingerprint
from m1_6_colab_gpu_smoke import cuda_memory, safe_load, timed_cuda
from m1_7_g1b_gpu_smoke import copy_backup, finite_optimizer, fingerprints, restore
from m1_7_g2_train import digest, write_new
from m1_7_g3a_eval import aggregate, confusion
from r1_dataset import read_manifest, require_hash
from r2_train import _checkpoint
from research_data import ResearchData


KIND = 'OUEM_RESEARCH_V5_CUDA_V1'
PLAN = 'b6711c33eafed51dd8d42040abf13e3a52601d5be54b08479217ac7e043c8692'
REVIEW = 'ccda53e9dac2c40fda6be718b9fc8a57dc35e6210fcc271a5a5533e390456dbc'
FREEZE = 'd7a72bdf1c792678cee0030c72dded18d3cdd6639a800c12ebafc63ba3fa1978'
CANDIDATES = 'bf71faf1ab512a326804f808aeff541f8e1363c8dece6d2250c6ebe7cbece0db'
IDENTITY = 'a645e831c4214cab913506e01fe9160f5272cf8aed1a6b00bbbf35964cd3b6d7'
INITIALIZER = 'f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036'
VAL = (1241, 205, 461, 1575, 1137, 1609, 4328, 22, 5014, 5003, 1950, 618, 1207, 2006, 3279, 802, 2861, 3245)
EVAL = (0, 70, 140, 210, 280)
SAVE = (1, *EVAL[1:])
ADAMW = dict(lr=1e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)


def make_schedule(data):
    train = sorted(i for i, row in data.samples.items() if row['split'] == 'train')
    if len(train) != 70 or set(data.samples) != set(train).union(VAL):
        raise ValueError('Expected fixed Train70 and Val18')
    if (614 not in train or Counter(len(data.crop_positions[i]) for i in train) != {1: 1, 2: 69}
            or len(data.crop_positions[614]) != 1
            or any(len(set(data.crop_positions[i])) != len(data.crop_positions[i]) for i in train)):
        raise ValueError('Expected 139 CLEAR windows, including single-window source614')
    if any(data.samples[i]['split'] != 'val' or len(data.crop_positions[i]) != 1 for i in VAL):
        raise ValueError('Fixed Val18 required')

    def entry(i, cycle):
        windows = sorted(data.crop_positions[i])
        x, y = windows[cycle % len(windows)]
        return dict(image_id=i, oam_id=data.samples[i]['oam_id'], crop_xywh=[x, y, 224, 224])

    rng = random.Random(0)
    updates = []
    for cycle in range(4):
        order = train.copy()
        rng.shuffle(order)
        for i in order:
            updates.append(dict(step=len(updates)+1, **entry(i, cycle)))
    return dict(version=1, seed=0, train=updates, val=[entry(i, 0) for i in VAL])


def evaluate(model, data, entries, fixed, device, panel=None):
    if entries != fixed:
        raise ValueError('Fixed validation identity mismatch')
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(param, param.requires_grad) for param in model.parameters()]
    captured = []
    before = fingerprint(model.state_dict())
    hook = model.head.layers.register_forward_hook(lambda _m, _i, output: captured.append(output))
    rows = []
    try:
        model.eval()
        with torch.inference_mode():
            for item in entries:
                captured.clear()
                x, y, _ = data.patch(item, 'val')
                x, y = x.to(device), y.to(device)
                probability, loss = model(x, y)
                if len(captured) != 1:
                    raise ValueError('Expected exactly one head output')
                logits = captured.pop()
                if logits.shape != (1, 2, *y.shape[-2:]) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid validation logits')
                ce = torch.nn.functional.cross_entropy(logits, y)
                p = logits.softmax(1)
                if probability.shape != p.shape or loss.ndim != 0 or not torch.allclose(probability, p, rtol=1e-5, atol=1e-6) or not torch.allclose(loss, ce, rtol=1e-5, atol=1e-7):
                    raise ValueError('Official CE or probability mismatch')
                pred, truth = logits.argmax(1)[0].cpu().numpy(), y[0].cpu().numpy()
                rows.append(dict(item, ce=ce.item(), **confusion(pred, truth)))
                if panel:
                    panel(item, x[0].cpu(), truth, pred, p[0, 1].cpu().numpy())
    finally:
        hook.remove()
        for module, mode in modes:
            module.training = mode
        for param, flag in gradients:
            param.requires_grad_(flag)
    if fingerprint(model.state_dict()) != before:
        raise ValueError('Evaluation changed model state')
    return dict(windows=rows, **aggregate(rows),
                background_only=[r for r in rows if r['tp']+r['fn'] == 0],
                caveats=['Val2006 linear positives', 'Val802 planting-row ambiguity',
                         'Val18 has no Japanese positive-tree validation'])


def update(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    probability, loss = model(x, y)
    if probability.shape != (1, 2, 224, 224) or loss.ndim != 0 or not torch.isfinite(loss) or not torch.isfinite(probability).all():
        raise ValueError('Invalid training probability or loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or nonfinite gradients')
    optimizer.step()
    finite_parameters(model)
    finite_optimizer(optimizer, torch)
    return float(loss.detach())


def validate_checkpoint(state, binding, schedule):
    completed = state.get('completed_steps')
    if (state.get('kind') != KIND or state.get('binding') != binding
            or type(completed) is not int or completed not in SAVE
            or state.get('schedule') != schedule or digest(state['schedule']) != binding['schedule_sha256']):
        raise ValueError('Checkpoint contract mismatch')
    next_item = schedule['train'][completed] if completed < 280 else None
    if state.get('next_item') != next_item:
        raise ValueError('Checkpoint next window mismatch')
    if not isinstance(state.get('backup_checkpoint'), str) or type(state.get('deadline')) not in (int, float) or not math.isfinite(state['deadline']):
        raise ValueError('Checkpoint backup/deadline mismatch')
    if not {'model', 'optimizer', 'python_rng', 'cpu_rng', 'cuda_rng', 'fingerprints'} <= state.keys():
        raise ValueError('Incomplete checkpoint state')
    return completed


def continuation_claim(state, checkpoint, sha):
    if checkpoint.resolve() != Path(state['backup_checkpoint']).resolve():
        raise ValueError('Resume requires the checkpoint in its original verified backup directory')
    write_new(checkpoint.parent/'continuation.json',
              dict(checkpoint_sha256=sha, next_item=state['next_item']))


def reserve(output, backup, protected):
    output, backup = output.resolve(), backup.resolve()
    repo = Path(__file__).resolve().parents[1]
    for destination, other in [(output, backup), (backup, output)]:
        for path in (repo, other, *protected):
            path = path.resolve()
            if destination == path or destination in path.parents or path in destination.parents:
                raise ValueError('Output/backup overlap with another destination or input')
        if destination.exists():
            raise FileExistsError(destination)
    output.mkdir(parents=True, exist_ok=False)
    backup.mkdir(parents=True, exist_ok=False)


def save_panel(path, rgb, truth, prediction, probability):
    gray = lambda a: np.repeat(a[:, :, None], 3, axis=2)
    panels = [np.rint(rgb.numpy().transpose(1, 2, 0)*255).astype(np.uint8),
              gray((truth*255).astype(np.uint8)), gray(np.rint(probability*255).astype(np.uint8)),
              gray(((prediction == 1) & (truth == 0)).astype(np.uint8)*255),
              gray(((prediction == 0) & (truth == 1)).astype(np.uint8)*255)]
    with path.open('xb') as stream:
        Image.fromarray(np.concatenate(panels, axis=1)).save(stream, format='PNG')


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA runtime required')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.vram_bytes <= 0:
        raise ValueError('Explicit positive time and VRAM ceilings required')
    import importlib.metadata
    import satlaspretrain_models as satlas
    import torchvision

    version = importlib.metadata.version('satlaspretrain-models')
    if version != '0.3.1':
        raise ValueError('SatlasPretrain 0.3.1 required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    gpu = torch.cuda.get_device_properties(0)
    environment = dict(python=platform.python_version(), torch=str(torch.__version__),
                       torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(), satlas=version, gpu=gpu.name,
                       total_vram=gpu.total_memory, devices=torch.cuda.device_count())
    deadline = time.time()+args.seconds
    torch.cuda.reset_peak_memory_stats()

    def budget():
        if time.time() >= deadline or torch.cuda.max_memory_reserved() > args.vram_bytes:
            raise RuntimeError('Time or reserved VRAM ceiling exceeded')

    require_hash(args.freeze, FREEZE)
    require_hash(args.initializer, INITIALIZER)
    data = ResearchData(args.dataset, PLAN, args.review, REVIEW,
                        candidates=args.candidates, candidates_sha256=CANDIDATES)
    freeze = read_manifest(args.freeze)
    if data.identity != freeze['identity'] or digest(data.identity) != IDENTITY:
        raise ValueError('Frozen v5 identity mismatch')
    schedule = make_schedule(data)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise ValueError('Clean checkout required to bind execution to a Git commit')
    binding = dict(inputs=data.identity, freeze_sha256=FREEZE, initializer_sha256=INITIALIZER,
                   schedule_sha256=digest(schedule), config=dict(seed=0, batch=1, precision='FP32',
                   scheduler=None, adamw=ADAMW, seconds=args.seconds, vram_bytes=args.vram_bytes),
                   environment=environment, commit=commit)
    state = safe_load(args.resume, args.resume_sha256) if args.resume else None
    completed = validate_checkpoint(state, binding, schedule) if state else 0
    if args.verify_restore and not state:
        raise ValueError('Restore verification requires a checkpoint')
    if state and not args.verify_restore:
        if completed == 280 or not args.execute:
            raise ValueError('Continuation requires an unfinished checkpoint and --execute')
        deadline = state['deadline']
    budget()
    if not args.execute and not args.verify_restore:
        print(dict(status='PREFLIGHT_PASS', binding=binding))
        return
    protected = [args.dataset, args.review, args.freeze, args.candidates, args.initializer]
    if state:
        protected.append(args.resume.parent)
    if not args.verify_restore:
        reserve(args.output, args.backup, protected)
    record = dict(status='PARTIAL', completed_steps=completed, binding=binding)
    try:
        if state:
            schedule = state['schedule']
        if not args.verify_restore:
            write_new(args.output/'schedule.json', schedule)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        weights = safe_load(args.initializer, INITIALIZER)
        model = satlas.Model(num_channels=3, multi_image=False, backbone=satlas.Backbone.SWINB,
                             fpn=True, head=satlas.Head.SEGMENT, num_categories=2, weights=weights)
        del weights
        model.to(device='cuda', dtype=torch.float32).requires_grad_(True).train()
        optimizer = torch.optim.AdamW(model.parameters(), **ADAMW)
        if state:
            restored = restore(model, optimizer, state, torch)
            budget()
            if args.verify_restore:
                print(dict(status='RESTORE_PASS', fingerprints=restored, next_item=state['next_item']))
                return
            continuation_claim(state, args.resume, args.resume_sha256)

        def assess(step):
            directory = args.output/f'val_{step:03d}'
            directory.mkdir()

            def panel(item, *arrays):
                budget()
                save_panel(directory/f"{item['image_id']}.png", *arrays)

            result = evaluate(model, data, schedule['val'], fixed_val, 'cuda', panel)
            write_new(directory/'metrics.json', result)
            budget()

        fixed_val = schedule['val']
        budget()
        if not completed:
            assess(0)
        for item in schedule['train'][completed:]:
            budget()
            x, y, _ = data.patch(item, 'train')
            x, y = x.cuda(), y.cuda()
            budget()
            loss, seconds = timed_cuda(torch, lambda: update(model, optimizer, x, y))
            step = item['step']
            record['completed_steps'] = step
            write_new(args.output/f'step_{step:03d}.json',
                      dict(item, loss=loss, seconds=seconds, memory=cuda_memory(torch)))
            budget()
            if step in EVAL:
                assess(step)
            if step in SAVE:
                path = args.output/f'checkpoint_{step:03d}.pth'
                destination = args.backup.resolve()/f'step_{step:03d}'
                payload = dict(kind=KIND, binding=binding, schedule=schedule, completed_steps=step,
                               next_item=schedule['train'][step] if step < 280 else None,
                               backup_checkpoint=str(destination/path.name), deadline=deadline,
                               model=model.state_dict(), optimizer=optimizer.state_dict(),
                               python_rng=random.getstate(), cpu_rng=torch.get_rng_state(),
                               cuda_rng=torch.cuda.get_rng_state_all(),
                               fingerprints=fingerprints(model, optimizer, torch))
                sha = _checkpoint(path, payload)
                write_new(args.output/f'backup_{step:03d}.json', copy_backup(path, sha, destination))
                budget()
        record['status'] = 'COMPLETE'
    except BaseException as exc:
        record['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if not args.verify_restore:
            write_new(args.output/'result.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'review', 'freeze', 'candidates', 'initializer', 'output', 'backup'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--vram-bytes', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--verify-restore', action='store_true')
    args = parser.parse_args()
    if bool(args.resume) != bool(args.resume_sha256):
        parser.error('--resume and --resume-sha256 must be supplied together')
    run(args)


if __name__ == '__main__':
    main()
