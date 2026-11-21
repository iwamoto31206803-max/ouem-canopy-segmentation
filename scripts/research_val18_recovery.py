"""CPU-only reconstruction of individual D5 assessments from reviewed local inputs."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
TRAINER = '30c3d9a63543550bfe7321f5a7dad11e646e3b94'
RUN = 'd5-v5-main280-01'
SCHEDULE = '5c3a1483e1087645e2ae472d993aae520102106da6d55a66a67ca1469661c601'
STEP280 = (1066503375, 'eeabeb671e0f0adeffd75267f06de752987f3ab2e51651875febd5a46cd6790b',
           '57dbefa89730adb270f17120079ecc3799f4da73b834be838a015d724d981835')
STEPS = (0, 70, 140, 210, 280)
VAL = (1241, 205, 461, 1575, 1137, 1609, 4328, 22, 5014, 5003, 1950, 618, 1207, 2006, 3279, 802, 2861, 3245)
PIXELS = 224**2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, stderr=subprocess.DEVNULL)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    require(not git('status', '--porcelain'), 'Clean evaluation checkout required')
    # Verify the historical scripts, including transitive local helpers. BOM and
    # content changes fail; Windows Git checkout line endings are normalized.
    paths = git('ls-tree', '-r', '--name-only', TRAINER, 'scripts').decode().splitlines()
    pins = {}
    for name in paths:
        if not name.endswith('.py'):
            continue
        historical = git('show', f'{TRAINER}:{name}').replace(b'\r\n', b'\n')
        current = (REPO / name).read_bytes().replace(b'\r\n', b'\n')
        require(historical == current, 'Historical source drift')
        pins[name] = hashlib.sha256(current).hexdigest()
    require({'scripts/research_cuda.py', 'scripts/research_data.py',
             'scripts/m1_7_g3a_eval.py'} <= pins.keys(), 'Missing historical sources')
    return dict(trainer_commit=TRAINER, tool_commit=git('rev-parse', 'HEAD').decode().strip(),
                tool_sha256=sha(Path(__file__)), historical_python_sha256_lf=pins)


def entries_for(schedule, pilot, digest):
    entries = schedule['val']
    require(digest(schedule) == SCHEDULE and tuple(e['image_id'] for e in entries) == VAL
            and len(schedule['train']) == 280, 'Fixed schedule mismatch')
    require(all(type(e['image_id']) is int and isinstance(e['oam_id'], str) and e['oam_id']
                and len(e['crop_xywh']) == 4 and all(type(v) is int for v in e['crop_xywh'])
                and min(e['crop_xywh'][:2]) >= 0 and e['crop_xywh'][2:] == [224, 224]
                for e in entries), 'Invalid Val window')
    return entries[:1] if pilot else entries


def validate_state(state, receipt, pins, step, data, schedule, cuda):
    from m1_7_g2_train import digest

    binding = state['binding']
    config = dict(seed=0, batch=1, precision='FP32', scheduler=None, adamw=cuda.ADAMW,
                  seconds=900.0, vram_bytes=12 * 1024**3)
    expected = dict(run_id=RUN, mode='main', inputs=data.identity, freeze_sha256=cuda.FREEZE,
                    initializer_sha256=cuda.INITIALIZER, schedule_sha256=SCHEDULE, commit=TRAINER)
    require(set(binding) == {*expected, 'lineage_id', 'config', 'environment'}
            and all(binding[k] == v for k, v in expected.items())
            and digest(binding['config']) == digest(config)
            and binding['environment']['satlas'] == '0.3.1', 'Historical binding mismatch')
    lineage = binding['lineage_id']
    require(isinstance(lineage, str) and re.fullmatch('[0-9a-f]{32}', lineage)
            and ('lineage_id' not in pins or pins['lineage_id'] == lineage), 'Lineage mismatch')
    require(cuda.validate_checkpoint(state, dict(expected, lineage_id=pins.get('lineage_id', lineage),
                                               config=binding['config'], environment=binding['environment']),
                                     schedule) == step, 'Wrong checkpoint step')
    segment = state['segment_id']
    elapsed = receipt['active_seconds']
    path = PurePosixPath(state['backup_checkpoint'])
    require(path.is_absolute() and '..' not in path.parts
            and path.parts[-3:] == (RUN, f'step_{step:03d}', f'checkpoint_{step:03d}.pth')
            and set(receipt) == {'checkpoint_sha256', 'segment_id', 'active_seconds'}
            and receipt['checkpoint_sha256'] == pins['sha256']
            and re.fullmatch('[0-9a-f]{32}', segment) and receipt['segment_id'] == segment
            and type(elapsed) in (int, float) and state['active_seconds'] <= elapsed < 900,
            'Runtime receipt or historical path mismatch')
    return dict(segment_id=segment, lineage_id=lineage,
                independent_lineage='VERIFIED' if 'lineage_id' in pins else 'UNVERIFIED',
                receipt_active_seconds=elapsed,
                historical_environment=binding['environment'])


def verified_load(path, expected, size, torch):
    with path.open('rb') as stream:
        if size is not None:
            require(os.fstat(stream.fileno()).st_size == size, 'Checkpoint size mismatch')
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected, 'Input SHA mismatch')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=True)


def model_for(args, data, schedule, cuda, torch, satlas):
    from m1_4_bounded_overfit import fingerprint
    from research_checkpoint_audit import pinned_json
    from satlaspretrain_models.model import adjust_state_dict_prefix

    start = time.monotonic()
    state = None
    provenance = {}
    if args.step:
        pins = pinned_json(args.pins, args.pins_sha256)
        required = {'step', 'size', 'sha256', 'receipt_sha256', 'evidence_ref'}
        require(required <= pins.keys() <= required | {'lineage_id'}
                and type(pins['step']) is int and pins['step'] == args.step
                and type(pins['size']) is int and pins['size'] > 0
                and all(isinstance(pins[k], str) and re.fullmatch('[0-9a-f]{64}', pins[k])
                        for k in ('sha256', 'receipt_sha256'))
                and isinstance(pins['evidence_ref'], str) and pins['evidence_ref'].strip(),
                'Reviewed checkpoint and receipt pins required')
        if args.step == 280:
            require(tuple(pins[k] for k in ('size', 'sha256', 'receipt_sha256')) == STEP280,
                    'Archived Step280 pins mismatch')
        receipt = pinned_json(args.receipt, pins['receipt_sha256'])
        state = verified_load(args.checkpoint, pins['sha256'], pins['size'], torch)
        provenance.update(pins=pins, pin_contract_sha256=args.pins_sha256)
        provenance.update(validate_state(state, receipt, pins, args.step, data, schedule, cuda))
    random.seed(0)
    torch.random.default_generator.manual_seed(0)
    weights = verified_load(args.initializer, cuda.INITIALIZER, None, torch)
    load_seconds = time.monotonic() - start
    start = time.monotonic()
    model = satlas.Model(num_channels=3, multi_image=False, backbone=satlas.Backbone.SWINB,
                         fpn=True, head=satlas.Head.SEGMENT, num_categories=2, weights=weights)
    model.to(device='cpu', dtype=torch.float32)
    if state is None:
        checks = {}
        for name, prefix, count in [('backbone', 'backbone.', 1), ('fpn', 'intermediates.0.', 0)]:
            official = adjust_state_dict_prefix(weights, name, prefix, count)
            actual = getattr(model, name).state_dict()
            require(official and official.keys() == actual.keys()
                    and all(torch.equal(actual[k], official[k]) for k in actual),
                    'Official component weights mismatch')
            checks[name] = 'EXACT_OFFICIAL_TENSORS'
        provenance.update(official_components=checks, original_head_equality='UNVERIFIED')
    else:
        saved = state['model']
        model_sha = state['fingerprints']['model']
        require(fingerprint(saved) == model_sha, 'Saved model fingerprint mismatch')
        model.load_state_dict(saved, strict=True)
        require(fingerprint(model.state_dict()) == model_sha, 'Loaded model fingerprint mismatch')
    provenance.update(model_fingerprint=fingerprint(model.state_dict()))
    return model, provenance, dict(load_seconds=load_seconds, construct_seconds=time.monotonic() - start)


def check_metrics(metrics, entries):
    from m1_7_g3a_eval import aggregate, count_metrics

    rows = metrics['windows']
    require(len(rows) == len(entries), 'Unexpected evaluation row count')
    for row, entry in zip(rows, entries):
        require(all(row[k] == v for k, v in entry.items())
                and math.isfinite(row['ce']) and row['ce'] >= 0, 'Row identity or CE mismatch')
        derived = count_metrics(*(row[k] for k in ('tp', 'fp', 'fn', 'tn')))
        require(derived['valid_pixels'] == PIXELS and all(row[k] == v for k, v in derived.items()),
                'Confusion totals or rates mismatch')
    require(metrics['ce_valid_pixels'] == len(entries) * PIXELS
            and all(metrics[k] == v for k, v in aggregate(rows).items()), 'Aggregate mismatch')


def peak_memory():
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('size', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
                (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set', 'peak_paged',
                'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
        counters = Counters()
        counters.size = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        require(read(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size), 'Memory probe failed')
        return dict(peak_working_set_bytes=counters.peak_working_set,
                    peak_private_commit_bytes=counters.peak_pagefile)
    if sys.platform.startswith('linux'):
        import resource
        return dict(peak_working_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    peak_private_commit_bytes='NOT_MEASURED')
    raise ValueError('No supported process memory probe')


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def worker(args, output, source):
    started = utc()
    try:
        require(source == sources(), 'Running sources changed')
        import torch
        import torchvision
        import satlaspretrain_models as satlas
        import research_cuda as cuda
        from m1_4_bounded_overfit import fingerprint
        from m1_7_g2_train import digest
        from research_checkpoint_audit import available_memory
        from PIL import Image

        version = importlib.metadata.version('satlaspretrain-models')
        require(version == '0.3.1', 'SatlasPretrain version mismatch')
        torch.set_num_threads(args.threads)
        memory = available_memory()
        require(memory >= args.min_ram_bytes, 'Insufficient available RAM')
        require(sha(args.freeze) == cuda.FREEZE and sha(args.initializer) == cuda.INITIALIZER,
                'Freeze or initializer SHA mismatch')
        data = cuda.ResearchData(args.dataset, cuda.PLAN, args.review, cuda.REVIEW,
                                 candidates=args.candidates, candidates_sha256=cuda.CANDIDATES)
        freeze = cuda.read_manifest(args.freeze)
        require(data.identity == freeze['identity'] and digest(data.identity) == cuda.IDENTITY,
                'Frozen v5 identity mismatch')
        schedule = cuda.make_schedule(data)
        entries = entries_for(schedule, args.pilot, digest)
        model, provenance, timings = model_for(args, data, schedule, cuda, torch, satlas)
        render_seconds = 0.0

        def panel(item, *arrays):
            nonlocal render_seconds
            require(item in entries, 'Unexpected panel identity')
            then = time.monotonic()
            cuda.save_panel(output / f"{item['image_id']}.png", *arrays)
            render_seconds += time.monotonic() - then

        then = time.monotonic()
        metrics = cuda.evaluate(model, data, entries, entries, 'cpu', panel)
        timings.update(evaluation_excluding_render_seconds=time.monotonic() - then - render_seconds,
                       render_seconds=render_seconds)
        require(fingerprint(model.state_dict()) == provenance['model_fingerprint'], 'Model state changed')
        check_metrics(metrics, entries)
        pngs = {f'{e["image_id"]}.png' for e in entries}
        require({p.name for p in output.iterdir()} == pngs, 'Panel inventory mismatch')
        for name in pngs:
            with Image.open(output / name) as image:
                require(image.format == 'PNG' and image.mode == 'RGB'
                        and image.size == (1120, 224), 'Invalid five-strip panel')
                image.load()
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'provenance.json', dict(
            provenance, **source, recovery_id=args.recovery_id, original_run=RUN, step=args.step,
            classification='RECOMPUTED_FROM_CHECKPOINT' if args.step else 'RECONSTRUCTED_STEP0',
            scope='PILOT_ONLY' if args.pilot else 'FIXED_VAL18_RECOMPUTATION',
            non_acceptance=args.pilot, val=entries, full_val=schedule['val'], schedule_sha256=SCHEDULE,
            dataset_identity=data.identity, freeze_sha256=cuda.FREEZE, initializer_sha256=cuda.INITIALIZER,
            device='cpu', precision='FP32', threads=args.threads, python=platform.python_version(),
            torch=str(torch.__version__), torchvision=str(torchvision.__version__), satlas=version,
            timings=timings, available_ram_before_load_bytes=memory, memory=peak_memory(),
            started_utc=started, finished_utc=utc()))
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='INCOMPLETE', error_type=type(exc).__name__))
        raise SystemExit(1) from None


def protected_destination(path, protected):
    require(not path.is_symlink(), 'Output symlink forbidden')
    path = path.resolve()
    require(path.parent.is_dir() and not path.exists()
            and not any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected),
            'Fresh private destination outside input and Git trees required')
    return path


def publish(output, copy, expected_names, pilot=False):
    observed = {p.name for p in output.iterdir() if p.is_file() and not p.is_symlink()}
    require(observed == expected_names and len(list(output.iterdir())) == len(expected_names),
            'Incomplete assessment')
    hashes = {name: sha(output / name) for name in sorted(expected_names)}
    write_json(output / 'sha256.json', hashes)
    index_sha = sha(output / 'sha256.json')
    require(json.loads((output / 'sha256.json').read_bytes()) == hashes
            and all(sha(output / name) == pin for name, pin in hashes.items()), 'Local readback mismatch')
    receipt = dict(status='LOCAL_EVIDENCE_COMPLETE', index_sha256=index_sha,
                   scope='PILOT_ONLY' if pilot else 'FIXED_VAL18_RECOMPUTATION', non_acceptance=pilot,
                   local_copy='NOT_REQUESTED', cloud='CLOUD_DURABILITY_UNVERIFIED')
    if copy:
        copy.mkdir()
        for name, pin in {**hashes, 'sha256.json': index_sha}.items():
            with (output / name).open('rb') as source, (copy / name).open('xb') as dest:
                shutil.copyfileobj(source, dest)
            require(sha(copy / name) == pin, 'Local copy SHA mismatch')
        receipt['local_copy'] = 'LOCAL_COPY_READBACK_PASS'
        write_json(copy / 'completion.json', receipt)
        require(json.loads((copy / 'completion.json').read_bytes()) == receipt, 'Copy receipt mismatch')
    write_json(output / 'completion.json', receipt)


def execute(args):
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    source = sources()
    inputs = [args.dataset, args.review, args.freeze, args.candidates, args.initializer]
    inputs += [p for p in (args.checkpoint, args.receipt, args.pins) if p is not None]
    require(args.dataset.is_dir() and all(p.is_file() for p in inputs[1:]), 'Expected local inputs')
    protected = [REPO, args.dataset.resolve(strict=True), *(p.resolve(strict=True).parent for p in inputs[1:])]
    output = protected_destination(args.output, protected)
    copy = protected_destination(args.local_copy, [*protected, output]) if args.local_copy else None
    for path in (output, copy):
        if path:
            require(shutil.disk_usage(path.parent).free >= args.min_disk_bytes, 'Insufficient output disk')
    output.mkdir()
    process = multiprocessing.get_context('spawn').Process(target=worker, args=(args, output, source))
    process.start()
    try:
        process.join(args.timeout_seconds)
        require(not process.is_alive(), 'CPU assessment timeout')
        require(process.exitcode == 0, 'CPU assessment failed')
        ids = VAL[:1] if args.pilot else VAL
        publish(output, copy, {'metrics.json', 'provenance.json', *(f'{i}.png' for i in ids)}, args.pilot)
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--assess', action='store_true')
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--step', type=int, choices=STEPS)
    for name in ('dataset', 'review', 'freeze', 'candidates', 'initializer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('checkpoint', 'receipt', 'pins', 'local-copy'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--pins-sha256')
    parser.add_argument('--recovery-id', required=True)
    for name in ('threads', 'timeout-seconds', 'min-ram-bytes', 'min-disk-bytes'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args(argv)
    if args.pilot and args.step is None:
        args.step = 280
    pins = (args.checkpoint, args.receipt, args.pins, args.pins_sha256)
    if (args.step is None or (args.pilot and args.step != 280)
            or (args.step != 0 and (not all(pins) or not re.fullmatch('[0-9a-f]{64}', args.pins_sha256)))
            or (args.step == 0 and any(pins))
            or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.recovery_id)
            or not 1 <= args.threads <= (os.cpu_count() or 1)
            or not 1 <= args.timeout_seconds <= 86400
            or args.min_ram_bytes < 8 * 1024**3 or args.min_disk_bytes < 16 * 1024**2):
        parser.error('Invalid mode, reviewed pins, recovery ID or CPU resource bounds')
    return args


def main():
    try:
        execute(parse())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'INCOMPLETE: {type(exc).__name__}', file=sys.stderr)
        return 1
    print('LOCAL_EVIDENCE_COMPLETE; CLOUD_DURABILITY_UNVERIFIED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
