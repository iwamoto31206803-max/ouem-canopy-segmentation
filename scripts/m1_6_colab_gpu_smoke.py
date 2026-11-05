"""Opt-in FP32 CUDA smoke, bounded to twenty total optimizer steps."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
CONFIG = dict(seed=0, batch_size=1, train_patch=256, val_patch=224, lr=1e-4,
              weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8, precision='FP32',
              architecture='Aerial_SwinB_SI/official-FPN/SEGMENT-2', threads=4)
KIND = 'OUEM_M1_6_GPU_SMOKE'
FILES = {'rgb.tif', 'canopy_fraction.tif', 'canopy_mask.tif', 'valid_fraction.tif'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True, help='Official initializer, read-only')
    result.add_argument('--manifest', type=Path, help='Transferred manifest; defaults to DATA/manifest.json')
    result.add_argument('--output', type=Path, required=True, help='Fresh run directory')
    result.add_argument('--max-steps', type=int, choices=range(1, 21), default=1, metavar='1..20',
                        help='Total completed step ceiling, including restored steps')
    result.add_argument('--resume', type=Path)
    result.add_argument('--resume-sha256')
    result.add_argument('--restore-only', action='store_true')
    result.add_argument('--allow-environment-change', action='store_true')
    return result


def reserve_output(args):
    if args.restore_only and args.resume is None:
        raise ValueError('--restore-only requires --resume')
    if bool(args.resume) != bool(args.resume_sha256):
        raise ValueError('--resume and --resume-sha256 must be supplied together')
    output = args.output.resolve()
    protected = [args.data.resolve(), args.checkpoint.resolve()]
    if args.manifest:
        protected.append(args.manifest.resolve())
    if args.resume:
        protected.append(args.resume.resolve())
    if output == REPO or REPO in output.parents:
        raise ValueError('Use an output outside the repository')
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError('Output overlaps read-only inputs')
    # Atomic reservation protects previous results, even on concurrent invocations.
    output.mkdir(parents=True, exist_ok=False)


def validate_inputs(data, supplied_manifest):
    from m1_3a_loader import load_patch
    from m1_2d_geometry_audit import sha
    authoritative_path = REPO / 'docs/implementation/M1_2D_PAIR_RESULT.json'
    authoritative = json.loads(authoritative_path.read_text())
    transferred = json.loads(supplied_manifest.read_text())
    if transferred['revision'] != authoritative['revision'] or transferred['dataset'] != authoritative['dataset']:
        raise ValueError('Dataset source revision mismatch')
    expected = authoritative['samples']
    if len(expected) != 2 or len(transferred['samples']) != 2:
        raise ValueError('Exactly two accepted samples required')
    if len({s['oam_id'] for s in expected}) != 2:
        raise ValueError('Train/Val source-group overlap')
    supplied = {s['image_id']: s for s in transferred['samples']}
    if set(supplied) != {661, 1972}:
        raise ValueError('Transferred sample identity mismatch')
    keys = ['image_id', 'oam_id', 'split', 'output_crs', 'output_transform', 'output_bounds', 'output_shape', 'output_hashes']
    patches = {}
    for sample in expected:
        if set(sample['output_hashes']) != FILES or any(sample[k] != supplied[sample['image_id']][k] for k in keys):
            raise ValueError('Transferred manifest/grid/hash mismatch')
        patches[sample['split']] = load_patch(data, sample, sample['split'], CONFIG[sample['split'] + '_patch'])
    return patches, dict(dataset=authoritative['dataset'], revision=authoritative['revision'],
        authoritative_manifest_sha256=sha(authoritative_path), transferred_manifest_sha256=sha(supplied_manifest),
        samples=expected)


def cuda_memory(torch):
    return dict(allocated_bytes=torch.cuda.memory_allocated(), max_allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(), max_reserved_bytes=torch.cuda.max_memory_reserved())


def timed_cuda(torch, function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def compatible(state, inputs, initializer, environment, git_sha, allow_change):
    if state.get('kind') != KIND or state.get('config') != CONFIG:
        raise ValueError('Unsupported checkpoint kind/config')
    step = state.get('completed_steps')
    if type(step) is not int or not 1 <= step <= 20:
        raise ValueError('Invalid completed step')
    if state['inputs'] != inputs or state['initializer'] != initializer:
        raise ValueError('Resume input/initializer provenance mismatch')
    changes = {key: [state['environment'].get(key), environment.get(key)] for key in environment
               if state['environment'].get(key) != environment.get(key)}
    if state['git_sha'] != git_sha:
        changes['git_sha'] = [state['git_sha'], git_sha]
    if changes and not allow_change:
        raise ValueError('Resume environment/commit differs; explicit --allow-environment-change required')
    return changes


def safe_load(path, expected_hash):
    import torch
    from m1_1_satlas_smoke import checkpoint_digest
    checkpoint_digest(path, expected_hash)
    return torch.load(path, map_location='cpu', weights_only=True)


def save_state(path, payload):
    import torch
    from m1_2d_geometry_audit import sha
    # Open exclusively: no prior trained artifact can be overwritten.
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return sha(path)


def run(args, record):
    record['stage'] = 'environment'
    import torch
    record['environment'] = dict(python=platform.python_version(), torch=str(torch.__version__),
                                 cuda_build=torch.version.cuda, platform=platform.platform())
    record['cuda_available'] = torch.cuda.is_available()
    record['git_sha'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    record['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip())
    if not record['cuda_available']:
        record['status'] = 'CUDA_NOT_VERIFIED'
        raise RuntimeError('CUDA is unavailable; select a GPU runtime and inspect the installed torch build')
    properties = torch.cuda.get_device_properties(0)
    record['environment'].update(gpu=properties.name, total_vram_bytes=properties.total_memory,
        device_count=torch.cuda.device_count(), capability=list(torch.cuda.get_device_capability(0)))
    torch.cuda.set_device(0)
    record['stage'] = 'imports'
    import satlaspretrain_models as satlas
    import inspect
    import satlaspretrain_models.model as official_model
    import satlaspretrain_models.models.heads as official_heads
    from m1_1_satlas_smoke import validate_output, gradient_evidence
    from m1_4_bounded_overfit import finite_parameters, metrics, fingerprint
    from m1_2d_geometry_audit import sha
    record['environment'].update({p: version(p) for p in
        ['torchvision', 'satlaspretrain-models', 'numpy', 'rasterio', 'pyproj']})
    if record['environment']['satlaspretrain-models'] != '0.3.1':
        raise ValueError('This smoke requires satlaspretrain-models 0.3.1')
    if record['git_dirty']:
        raise ValueError('Use a clean, reviewed commit for the GPU experiment')
    record['official_source_hashes'] = {name: sha(Path(inspect.getfile(module)))
        for name, module in [('model', official_model), ('heads', official_heads)]}
    record['stage'] = 'input_validation'
    patches, inputs = validate_inputs(args.data, args.manifest or args.data / 'manifest.json')
    record['input_tensors'] = {split: dict(rgb_shape=list(x.shape), rgb_dtype=str(x.dtype),
        rgb_range=[x.min().item(), x.max().item()], target_shape=list(y.shape), target_dtype=str(y.dtype),
        labels=y.unique().tolist(), crop_origin=[0, 0], valid_fraction_range=[
            float(crop['valid_fraction.tif'].min()), float(crop['valid_fraction.tif'].max())])
        for split, (x, y, crop) in patches.items()}
    initializer = json.loads((REPO / 'docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    from m1_1_satlas_smoke import checkpoint_digest
    checkpoint_digest(args.checkpoint, initializer['sha256'])
    record.update(config=CONFIG, inputs=inputs, initializer=initializer, max_steps=args.max_steps)
    torch.set_num_threads(CONFIG['threads'])
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    record['runtime_flags'] = dict(tf32=False, cudnn_benchmark=False,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), amp=False)
    record['environment']['runtime_flags'] = record['runtime_flags']
    record['environment']['cudnn_version'] = torch.backends.cudnn.version()
    torch.cuda.reset_peak_memory_stats()
    record['stage'] = 'model_construction'
    state = None
    completed = 0
    if args.resume:
        state = safe_load(args.resume, args.resume_sha256)
        record['environment_changes'] = compatible(state, inputs, initializer, record['environment'],
                                                   record['git_sha'], args.allow_environment_change)
        completed = state['completed_steps']
        if not args.restore_only and args.max_steps <= completed:
            raise ValueError('max-steps must exceed restored step count')
        weights = None
    else:
        weights = safe_load(args.checkpoint, initializer['sha256'])
        if not isinstance(weights, dict) or not weights or not all(isinstance(v, torch.Tensor) for v in weights.values()):
            raise ValueError('Official checkpoint must be a flat tensor state dict')
    started = time.perf_counter()
    model = satlas.Model(num_channels=3, multi_image=False, backbone=satlas.Backbone.SWINB,
                         fpn=True, head=satlas.Head.SEGMENT, num_categories=2, weights=weights)
    del weights
    if state:
        model.load_state_dict(state['model'], strict=True)
    model = model.to(device='cuda', dtype=torch.float32)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'],
                                 betas=tuple(CONFIG['betas']), eps=CONFIG['eps'])
    if state:
        optimizer.load_state_dict(state['optimizer'])
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device-count mismatch')
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        observed = dict(model=fingerprint(model.state_dict()), optimizer=fingerprint(optimizer.state_dict()),
                        cpu_rng=fingerprint(torch.get_rng_state()), cuda_rng=fingerprint(torch.cuda.get_rng_state_all()))
        if observed != state['fingerprints']:
            raise ValueError('Restored model/optimizer/RNG state fingerprints differ')
        record['restore_status'] = 'GPU_CHECKPOINT_STATE_RESTORED'
        record['restored_checkpoint_sha256'] = args.resume_sha256
        record['initial_eval'] = state['initial_eval']
        del state
    torch.cuda.synchronize()
    record['construction_restore_seconds'] = time.perf_counter() - started
    record['memory_after_construction'] = cuda_memory(torch)
    record['initialization'] = 'restored trained state' if args.resume else 'official strict backbone/FPN; new Upsample/Head'
    if args.restore_only:
        record.update(status='GPU_CHECKPOINT_RESTORE_PASS', completed_steps=completed)
        return
    patches = {split: (x.to('cuda'), y.to('cuda')) for split, (x, y, _) in patches.items()}
    def evaluate():
        model.eval()
        result = {}
        with torch.no_grad():
            for split, (images, target) in patches.items():
                probability, loss = model(images, target)
                validate_output(torch, probability, loss, images.shape[-1])
                result[split] = dict(ce=loss.item(), **metrics(probability.argmax(1), target))
        torch.cuda.synchronize()
        return result
    record['stage'] = 'initial_eval'
    record['before_segment_eval'] = evaluate()
    if not args.resume:
        record['initial_eval'] = record['before_segment_eval']
    record['steps'] = []
    for step in range(completed + 1, args.max_steps + 1):
        entry = dict(step=step)
        record['steps'].append(entry)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        images, target = patches['train']
        record['stage'] = 'forward_loss'
        (probability, loss), entry['forward_seconds'] = timed_cuda(torch, lambda: model(images, target))
        validate_output(torch, probability, loss, 256)
        entry['loss'] = loss.item()
        entry['memory_after_forward'] = cuda_memory(torch)
        record['stage'] = 'backward'
        _, entry['backward_seconds'] = timed_cuda(torch, loss.backward)
        entry['gradients'] = {c: gradient_evidence(torch, getattr(model, c)) for c in ['backbone', 'fpn', 'upsample', 'head']}
        representatives = {}
        for component, evidence in entry['gradients'].items():
            parameter = dict(getattr(model, component).named_parameters())[evidence['example']]
            representatives[component] = (parameter, parameter.detach().clone())
        entry['memory_after_backward'] = cuda_memory(torch)
        record['stage'] = 'optimizer_step'
        _, entry['optimizer_seconds'] = timed_cuda(torch, optimizer.step)
        finite_parameters(model)
        for value in optimizer.state.values():
            if not all(torch.isfinite(v).all() for v in value.values() if isinstance(v, torch.Tensor)):
                raise ValueError('Nonfinite optimizer state')
        entry['updated'] = {c: not torch.equal(p.detach(), before) for c, (p, before) in representatives.items()}
        if not optimizer.state or not all(entry['updated'].values()):
            raise ValueError('Active parameter update or optimizer state creation failed')
        entry['memory_after_step'] = cuda_memory(torch)
        completed = step
        record['completed_steps'] = completed
        if step == 1:
            record['optimizer_step_status'] = 'GPU_OPTIMIZER_STEP_PASS'
        print(json.dumps(dict(step=step, loss=entry['loss'])), flush=True)
        del probability, loss, representatives
    record['stage'] = 'final_eval'
    record['final_eval'] = evaluate()
    record['stage'] = 'checkpoint_save'
    model_state, optimizer_state = model.state_dict(), optimizer.state_dict()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    fingerprints = dict(model=fingerprint(model_state), optimizer=fingerprint(optimizer_state),
                        cpu_rng=fingerprint(cpu_rng), cuda_rng=fingerprint(cuda_rng))
    payload = dict(kind=KIND, config=CONFIG, completed_steps=completed, model=model_state,
        optimizer=optimizer_state, cpu_rng=cpu_rng, cuda_rng=cuda_rng, git_sha=record['git_sha'],
        environment=record['environment'], initializer=initializer, inputs=inputs, fingerprints=fingerprints,
        initial_eval=record['initial_eval'])
    path = args.output / 'gpu_smoke.pth'
    checkpoint_hash, record['save_seconds'] = timed_cuda(torch, lambda: save_state(path, payload))
    record['saved_checkpoint'] = dict(filename=path.name, sha256=checkpoint_hash, bytes=path.stat().st_size,
                                      fingerprints=fingerprints)
    record['status'] = 'GPU_SMOKE_STEPS_PASS_RESTORE_PENDING'
    record['memory_final'] = cuda_memory(torch)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        reserve_output(args)
    except (ValueError, FileExistsError) as error:
        parser().error(str(error))
    record = dict(status='NOT_VERIFIED', stage='startup', utc=datetime.now(timezone.utc).isoformat(),
                  config=CONFIG, max_steps=args.max_steps, restore_only=args.restore_only)
    started = time.perf_counter()
    code = 0
    try:
        run(args, record)
    except Exception as error:
        record['exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        if record['status'] != 'CUDA_NOT_VERIFIED':
            record['status'] = 'FAIL_STOPPED'
        try:
            import torch
            if torch.cuda.is_available():
                record['memory_at_failure'] = cuda_memory(torch)
        except Exception:
            pass
        code = 2 if record['status'] == 'CUDA_NOT_VERIFIED' else 1
    record['total_seconds'] = time.perf_counter() - started
    (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
