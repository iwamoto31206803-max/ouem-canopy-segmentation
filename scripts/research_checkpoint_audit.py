"""Offline inspection of the archived research-v5 Step280 serialized state."""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
import subprocess
import sys

import torch

from m1_4_bounded_overfit import fingerprint
from m1_7_g2_train import digest
from research_cuda import ADAMW, FREEZE, IDENTITY, INITIALIZER, KIND, VAL, validate_checkpoint


COMMIT = '30c3d9a63543550bfe7321f5a7dad11e646e3b94'
RUN = 'd5-v5-main280-01'
STEP = 280
SIZE = 1066503375
SHA = 'eeabeb671e0f0adeffd75267f06de752987f3ab2e51651875febd5a46cd6790b'
SCHEDULE = '5c3a1483e1087645e2ae472d993aae520102106da6d55a66a67ca1469661c601'
SEGMENT = '6988945ea1c24149be6dc29eed591f34'
ELAPSED = 310.096483756
SECONDS = 900.0
VRAM_BYTES = 12 * 1024**3
ENV_KEYS = {'python', 'torch', 'torchvision', 'cuda', 'cudnn', 'satlas', 'gpu', 'total_vram', 'devices'}
FIELDS = dict(model='model', optimizer='optimizer', python_rng='python_rng',
              cpu_rng='cpu_rng', cuda_rng='cuda_rng')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, expected):
    with path.open('rb') as stream:
        raw = stream.read(1024**2 + 1)
    require(len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == expected,
            'JSON size or SHA mismatch')

    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'Duplicate JSON key')
        return result

    def invalid_constant(_):
        raise ValueError('Non-finite JSON constant')

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    require(isinstance(value, dict), 'Expected JSON object')
    return value


def available_memory():
    if sys.platform == 'win32':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                        ('available', ctypes.c_ulonglong), ('remaining', ctypes.c_ulonglong * 5)]
        state = Memory()
        state.length = ctypes.sizeof(state)
        require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)), 'RAM probe failed')
        return state.available
    if sys.platform.startswith('linux'):
        with open('/proc/meminfo', encoding='ascii') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    raise ValueError('No supported available-RAM probe')


def load_checkpoint(path, size, sha):
    require(type(size) is int and size == SIZE and sha == SHA, 'Unexpected checkpoint pin')
    with path.open('rb') as stream:
        require(os.fstat(stream.fileno()).st_size == size, 'Checkpoint size mismatch')
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == sha, 'Checkpoint SHA mismatch')
        stream.seek(0)
        return torch.load(stream, map_location='cpu', weights_only=True)


def verify_binding(state, receipt, expected):
    require(isinstance(state, dict), 'Expected checkpoint mapping')
    require(set(expected) >= {'lineage_id', 'seconds', 'vram_bytes'}
            and set(expected) <= {'lineage_id', 'seconds', 'vram_bytes', 'environment',
                                  'parent', 'backup_checkpoint'}, 'Invalid expected contract keys')
    lineage = expected['lineage_id']
    seconds, vram = expected['seconds'], expected['vram_bytes']
    require(isinstance(lineage, str) and re.fullmatch('[0-9a-f]{32}', lineage)
            and type(seconds) in (int, float) and seconds == SECONDS
            and type(vram) is int and vram == VRAM_BYTES, 'Invalid independent contract')
    binding = state['binding']
    require(set(binding) == {'run_id', 'lineage_id', 'mode', 'inputs', 'freeze_sha256',
                             'initializer_sha256', 'schedule_sha256', 'config', 'environment',
                             'commit'}, 'Unexpected binding keys')
    require(isinstance(binding['inputs'], dict) and digest(binding['inputs']) == IDENTITY,
            'Frozen input identity mismatch')
    fixed = dict(run_id=RUN, lineage_id=lineage, mode='main', freeze_sha256=FREEZE,
                 initializer_sha256=INITIALIZER, schedule_sha256=SCHEDULE, commit=COMMIT)
    require(all(binding[key] == value for key, value in fixed.items()), 'Fixed binding mismatch')
    config = dict(seed=0, batch=1, precision='FP32', scheduler=None, adamw=ADAMW,
                  seconds=SECONDS, vram_bytes=VRAM_BYTES)
    require(digest(binding['config']) == digest(config), 'Training configuration mismatch')
    env = binding['environment']
    require(isinstance(env, dict) and set(env) == ENV_KEYS and env['satlas'] == '0.3.1'
            and all(type(env[k]) is int and env[k] > 0 for k in ('total_vram', 'devices', 'cudnn'))
            and all(isinstance(env[k], str) and env[k] for k in ENV_KEYS - {'total_vram', 'devices', 'cudnn'}),
            'Malformed historical environment')
    if 'environment' in expected:
        require(digest(env) == digest(expected['environment']), 'Historical environment mismatch')
    schedule = state['schedule']
    require(digest(schedule) == SCHEDULE and len(schedule['train']) == STEP
            and [row['image_id'] for row in schedule['val']] == list(VAL), 'Schedule mismatch')
    # The external digest pins every schedule entry; no private product reads are needed.
    checked_binding = dict(fixed, inputs=binding['inputs'], config=binding['config'],
                           environment=expected.get('environment', env))
    require(validate_checkpoint(state, checked_binding, schedule) == STEP and state['kind'] == KIND
            and state['next_item'] is None, 'Expected completed terminal checkpoint')
    require(set(receipt) == {'checkpoint_sha256', 'segment_id', 'active_seconds'}
            and receipt['checkpoint_sha256'] == SHA
            and receipt['segment_id'] == state['segment_id'] == SEGMENT
            and type(receipt['active_seconds']) in (int, float)
            and state['active_seconds'] <= receipt['active_seconds'] == ELAPSED < seconds,
            'Runtime receipt mismatch')
    path = PurePosixPath(state['backup_checkpoint'])
    require(path.is_absolute() and '..' not in path.parts
            and path.parts[-3:] == (RUN, 'step_280', 'checkpoint_280.pth'), 'Invalid historical backup path')
    parent = state['parent']
    if parent is not None:
        require(isinstance(parent, dict) and set(parent) == {'segment_id', 'checkpoint_sha256', 'completed_steps'}
                and isinstance(parent['segment_id'], str) and re.fullmatch('[0-9a-f]{32}', parent['segment_id'])
                and parent['segment_id'] != SEGMENT
                and isinstance(parent['checkpoint_sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', parent['checkpoint_sha256'])
                and type(parent['completed_steps']) is int and parent['completed_steps'] in (1, 70, 140, 210),
                'Invalid parent segment')
    for key in ('parent', 'backup_checkpoint'):
        if key in expected:
            require(digest(state[key]) == digest(expected[key]), 'Historical lineage provenance mismatch')
    return {key: 'PASS' if key in expected else 'UNVERIFIED'
            for key in ('environment', 'parent', 'backup_checkpoint')}


def tensor(value):
    return (isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.layout == torch.strided and value.numel() > 0)


def verify_payload(state):
    model = state['model']
    require(isinstance(model, dict) and model
            and all(isinstance(k, str) and k and tensor(v)
                    and v.dtype in (torch.float32, torch.int64, torch.int32, torch.bool)
                    and torch.isfinite(v).all() for k, v in model.items()), 'Invalid model tensors')
    optimizer = state['optimizer']
    require(isinstance(optimizer, dict) and set(optimizer) == {'state', 'param_groups'}
            and isinstance(optimizer['param_groups'], list) and len(optimizer['param_groups']) == 1,
            'Invalid optimizer groups')
    group = optimizer['param_groups'][0]
    defaults = dict(amsgrad=False, maximize=False, foreach=None, capturable=False,
                    differentiable=False, fused=None, decoupled_weight_decay=True)
    require(isinstance(group, dict) and set(group) <= {'params', *ADAMW, *defaults}
            and all(digest(group[k]) == digest(v) for k, v in ADAMW.items())
            and all(digest(group[k]) == digest(defaults[k]) for k in defaults.keys() & group.keys()),
            'AdamW settings mismatch')
    ids, slots = group['params'], optimizer['state']
    require(isinstance(ids, list) and ids and all(type(i) is int for i in ids)
            and ids == list(range(len(ids))) and isinstance(slots, dict) and slots
            and all(type(i) is int and 0 <= i < len(ids) for i in slots), 'Invalid optimizer parameter IDs')
    signature = lambda t: (tuple(t.shape), t.dtype)
    model_shapes = Counter(signature(v) for v in model.values() if v.is_floating_point())
    slot_shapes = Counter()
    for slot in slots.values():
        require(isinstance(slot, dict) and set(slot) == {'step', 'exp_avg', 'exp_avg_sq'},
                'Invalid AdamW slot keys')
        step, m, v = (slot[k] for k in ('step', 'exp_avg', 'exp_avg_sq'))
        require(tensor(step) and step.ndim == 0 and step.dtype in (torch.float32, torch.float64)
                and math.isfinite(step.item()) and 1 <= step.item() <= STEP and step.item().is_integer(),
                'Invalid AdamW step counter')
        require(tensor(m) and tensor(v) and m.dtype == torch.float32 and signature(m) == signature(v)
                and torch.isfinite(m).all() and torch.isfinite(v).all() and (v >= 0).all(),
                'Invalid AdamW moments')
        slot_shapes[signature(m)] += 1
    require(not slot_shapes - model_shapes and len(ids) <= sum(model_shapes.values()),
            'Optimizer/model tensor signatures inconsistent')
    cpu, cuda, python = (state[k] for k in ('cpu_rng', 'cuda_rng', 'python_rng'))

    def byte_state(v):
        return tensor(v) and v.dtype == torch.uint8 and v.ndim == 1 and v.is_contiguous()

    require(byte_state(cpu) and isinstance(cuda, list)
            and len(cuda) == state['binding']['environment']['devices']
            and all(byte_state(v) and v.numel() >= 16 for v in cuda)
            and len({v.numel() for v in cuda}) == 1 and isinstance(python, tuple),
            'Invalid serialized RNG payload')
    torch.Generator(device='cpu').set_state(cpu)
    random.Random(0).setstate(python)
    fingerprints = state['fingerprints']
    require(isinstance(fingerprints, dict) and set(fingerprints) == set(FIELDS), 'Incomplete fingerprints')
    for field, key in FIELDS.items():
        require(fingerprint(state[key]) == fingerprints[field], 'Fingerprint mismatch: ' + field)


def audit(args):
    inputs = [args.checkpoint, args.receipt, args.contract]
    require(all(p.is_file() and not p.is_symlink() for p in inputs), 'Inputs must be local regular files')
    require(len({p.resolve() for p in inputs}) == len(inputs), 'Inputs must be distinct')
    require(args.output.suffix == '.json' and args.output.parent.is_dir(), 'Fresh JSON output required')
    # Exclusive creation also rejects existing files, symlinks and hardlink aliases.
    with args.output.open('x', encoding='utf-8') as output:
        report = dict(status='FAIL', device='cpu', python=platform.python_version(), torch=str(torch.__version__),
                      checkpoint_sha256=args.sha256, bytes=args.size, receipt_sha256=args.receipt_sha256,
                      contract_sha256=args.contract_sha256, gates={}, gpu_restore='NOT_RUN',
                      limitations=['No model or optimizer reconstruction', 'No CUDA RNG activation',
                                   'Fingerprints check internal consistency of serialized values',
                                   'Optimizer tensor signatures do not establish name-to-slot mapping',
                                   'Schedule verified by the independent digest, no v5 products inspected'])
        gate = 'tool_revision'
        try:
            repo = Path(__file__).resolve().parents[1]
            report['tool_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
            report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            report['gates'][gate] = 'PASS'
            gate = 'pinned_json'
            receipt = pinned_json(args.receipt, args.receipt_sha256)
            expected = pinned_json(args.contract, args.contract_sha256)
            report['gates'][gate] = 'PASS'
            if not expected.get('lineage_id'):
                report['gates']['independent_lineage'] = 'UNVERIFIED'
                report['status'] = 'INDEPENDENT_LINEAGE_UNVERIFIED'
                return False
            gate = 'resources'
            report['available_ram_bytes'] = available_memory()
            require(report['available_ram_bytes'] >= max(4 * 1024**3, 8 * SIZE),
                    'Insufficient available RAM')
            report['output_free_bytes'] = shutil.disk_usage(args.output.parent).free
            require(report['output_free_bytes'] >= 1024**2, 'Insufficient output space')
            report['gates'][gate] = 'PASS'
            gate = 'checkpoint_bytes_and_cpu_load'
            state = load_checkpoint(args.checkpoint, args.size, args.sha256)
            report['gates'][gate] = 'PASS'
            gate = 'binding_schedule_and_receipt'
            report['historical_provenance'] = verify_binding(state, receipt, expected)
            if report['historical_provenance']['environment'] == 'UNVERIFIED':
                report['historical_environment_status'] = 'HISTORICAL_ENVIRONMENT_UNVERIFIED'
            report.update(run_id=RUN, lineage_id=expected['lineage_id'], segment_id=SEGMENT,
                          completed_steps=STEP, schedule_sha256=SCHEDULE, receipt_active_seconds=ELAPSED)
            report['gates'][gate] = 'PASS'
            gate = 'payload_and_serialized_fingerprints'
            verify_payload(state)
            report['gates'][gate] = 'PASS'
            report['status'] = 'CPU_SERIALIZED_STATE_VERIFIED'
            return True
        except Exception as exc:
            report['gates'][gate] = 'FAIL'
            # Loader exceptions can include filenames or serialized values.
            report['error'] = type(exc).__name__
            return False
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'receipt', 'contract', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('sha256', 'receipt-sha256', 'contract-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--size', required=True, type=int)
    parser.add_argument('--trusted-local-checkpoint', required=True, action='store_true')
    args = parser.parse_args()
    if any(not re.fullmatch('[0-9a-f]{64}', v)
           for v in (args.sha256, args.receipt_sha256, args.contract_sha256)):
        parser.error('Expected lowercase SHA-256 pins')
    try:
        return 0 if audit(args) else 1
    except (OSError, ValueError):
        parser.exit(1, 'Audit could not open distinct inputs and a fresh private output.\n')


if __name__ == '__main__':
    sys.exit(main())
