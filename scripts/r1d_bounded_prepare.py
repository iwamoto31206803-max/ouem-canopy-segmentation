"""Fixed-four bounded conversion rehearsal. Real-data execution is fail-closed."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from r1b_campaign import Stop, ordinary


SCHEMA = 'ouem-r1d-four-conversion-v1'
GENERATION = 'zenodo-11617167-v1.0.0'
TARGETS = {717: 'train', 1054: 'train', 1241: 'val', 461: 'val'}
LIMITS = dict(write_bytes=64*1024**2, report_bytes=8*1024**2,
              product_bytes=2*1024**2, memory_bytes=2*1024**3,
              deadline_seconds=1800, free_bytes=256*1024**2)
REPO = Path(__file__).resolve().parents[1]
PROTECTED = [REPO, Path(r'C:\dev\ouem-data-cache'),
             Path(r'C:\dev\ouem-data-audits'), Path(r'C:\dev\ouem-model-cache')]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected):
    ordinary(path)
    if Path(path).stat().st_size > 65536:
        raise Stop('Fixed-four control file exceeds 64 KiB')
    data = Path(path).read_bytes()
    if digest(data) != expected:
        raise Stop('Input file SHA mismatch')
    from r1_dataset import read_manifest
    return read_manifest(path)


def relative_path(name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise Stop('Unsafe relative path')
    parts = name.split('/')
    if any(not part or part in ('.', '..') or part.endswith((' ', '.'))
           or part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
              *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}
           for part in parts):
        raise Stop('Unsafe relative path')
    return name


class Budget:
    """Single-writer append-only reservations. Failed reservations remain charged."""

    def __init__(self, root, limits):
        self.root = Path(root).absolute()
        self.limits = limits
        self.charged = 0
        self.report_charged = 0
        self.operations = {}

    @classmethod
    def create(cls, root, limits):
        budget = cls(root, limits)
        ordinary(budget.root)
        destination = budget.root.resolve()
        for source in PROTECTED:
            source = source.resolve()
            if destination == source or source in destination.parents or destination in source.parents:
                raise Stop('Protected output overlap')
        if budget.root.exists():
            raise Stop('Output must be new')
        if not budget.root.parent.is_dir():
            raise Stop('Existing output parent required')
        if shutil.disk_usage(budget.root.parent).free < limits['free_bytes']:
            raise Stop('Insufficient free disk')
        budget.root.mkdir()
        with (budget.root/'journal.jsonl').open('xb'):
            pass
        return budget

    @classmethod
    def retained(cls, root, limits):
        """Controller recovery for a final report after the worker has exited."""
        budget = cls(root, limits)
        ordinary(budget.root)
        with (budget.root/'journal.jsonl').open('rb') as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                    name = relative_path(entry['path'])
                    if entry['event'] == 'RESERVE':
                        if name.casefold() in budget.operations:
                            raise Stop('Duplicate retained reservation')
                        budget.charged += entry['charge']
                        budget.report_charged += entry['report_charge']
                        budget.operations[name.casefold()] = entry
                    elif entry['event'] == 'COMMIT':
                        reservation = budget.operations[name.casefold()]
                        if reservation.get('complete'):
                            raise Stop('Duplicate retained commit')
                        path = budget.root/name
                        ordinary(path)
                        data = path.read_bytes()
                        if len(data) != entry['bytes'] or digest(data) != entry['sha256']:
                            raise Stop('Committed file changed')
                        reservation['complete'] = True
                    else:
                        raise Stop('Unknown journal event')
                except Stop:
                    raise
                except (ValueError, KeyError, TypeError) as exc:
                    raise Stop('Retained journal requires reconciliation') from exc
        if budget.charged > limits['write_bytes'] or budget.report_charged > limits['report_bytes']:
            raise Stop('Retained budget exceeded')
        return budget

    def _journal(self, entry):
        data = encoded(entry) + b'\n'
        if len(data) > 512:
            raise Stop('Journal record exceeds reserved control space')
        ordinary(self.root/'journal.jsonl')
        with (self.root/'journal.jsonl').open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _payload(self, stream, data):
        written = stream.write(data)
        if written != len(data):
            raise Stop('Short write')
        stream.flush()
        os.fsync(stream.fileno())

    def write(self, name, data, *, report=False):
        relative_path(name)
        key = name.casefold()
        path = self.root/name
        ordinary(path)
        partial = path.with_name(path.name+'.partial')
        ordinary(partial)
        if key in self.operations or path.exists() or partial.exists():
            raise Stop('Duplicate/pre-existing output')
        # Two <=512-byte journal records are reserved before any disk operation.
        charge = len(data) + 1024
        report_charge = 1024 + (len(data) if report else 0)
        if (self.charged + charge > self.limits['write_bytes'] or
                self.report_charged + report_charge > self.limits['report_bytes']):
            raise Stop('Cumulative write/report budget exceeded')
        if shutil.disk_usage(self.root).free < charge:
            raise Stop('Insufficient free disk')
        reservation = dict(event='RESERVE', path=name, charge=charge,
                           report_charge=report_charge, bytes=len(data))
        self.charged += charge
        self.report_charged += report_charge
        self.operations[key] = reservation
        self._journal(reservation)
        path.parent.mkdir(parents=True, exist_ok=True)
        ordinary(path.parent)
        with partial.open('xb') as stream:
            self._payload(stream, data)
        actual = partial.read_bytes()
        if len(actual) != len(data) or digest(actual) != digest(data):
            raise Stop('Partial changed before publication')
        ordinary(path)
        # link() refuses an existing destination on Windows and POSIX.
        os.link(partial, path)
        partial.unlink()
        self._journal(dict(event='COMMIT', path=name, bytes=len(data), sha256=digest(data)))
        reservation['complete'] = True
        return digest(data)

    def summary(self):
        files = [p for p in self.root.rglob('*') if p.is_file()]
        return dict(charged_upper_bound=self.charged,
                    report_charged_upper_bound=self.report_charged,
                    retained_file_bytes=sum(p.stat().st_size for p in files),
                    unresolved=[v['path'] for v in self.operations.values() if not v.get('complete')])


def fixture_assets(image_id, label):
    """Generate only tiny synthetic bytes. No user-specified source paths are read."""
    import numpy as np
    from PIL import Image
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    group = f'synthetic-{image_id}'
    row = dict(image_id=image_id, oam_id=group, official_split='train',
               validation_fold=0 if TARGETS[image_id] == 'val' else 1,
               license='CC-BY 4.0', crs='EPSG:3395', bounds=[0, 0, 2, 2],
               height=20, width=20, meta=f'https://invalid.example/{group}_meta.json')
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', height=20, width=20, count=3,
                         dtype='uint8', crs=row['crs'], transform=from_origin(0, 2, .1, .1)) as dst:
            dst.write(np.full((3, 20, 20), 110, np.uint8))
        rgb = memory.read()
    annotation = np.zeros((20, 20, 3), np.uint8)
    if label == 'one':
        annotation[:, :, 0] = 9
    elif label == 'half':
        annotation[:10, :, 0] = 9
    buffer = io.BytesIO()
    Image.fromarray(annotation).save(buffer, format='PNG')
    stac = dict(id=group, bbox=[-1, -1, 1, 1],
                properties={'license': 'CC-BY-4.0', 'oam:producer_name': 'SYNTHETIC',
                            'providers': [dict(name='SYNTHETIC', roles=['licensor'])]},
                assets={'metadata': {'href': row['meta']}})
    return row, rgb, buffer.getvalue(), encoded(stac)


def synthetic_plan():
    samples = []
    for image_id, label in zip(TARGETS, ['zero', 'one', 'half', 'half']):
        _, rgb, annotation, stac = fixture_assets(image_id, label)
        samples.append(dict(image_id=image_id, split=TARGETS[image_id], label=label,
                            expected_shape=[4, 4], image_sha256=digest(rgb),
                            annotation_sha256=digest(annotation), stac_sha256=digest(stac),
                            license='CC-BY-4.0'))
    return dict(schema=SCHEMA, mode='SYNTHETIC_ONLY', source_generation=GENERATION,
                samples=samples, limits=dict(LIMITS))


def validate_plan(plan, plan_sha, stub):
    if not isinstance(plan, dict):
        raise Stop('Plan must be an object')
    if plan.get('mode') != 'SYNTHETIC_ONLY':
        raise Stop('REAL_EXECUTION_DISABLED: native network isolation and G3/G4 execution approvals unresolved')
    if set(plan) != {'schema', 'mode', 'source_generation', 'samples', 'limits'}:
        raise Stop('Unknown plan fields/source paths')
    if plan['schema'] != SCHEMA or plan['source_generation'] != GENERATION:
        raise Stop('Plan schema/source generation mismatch')
    if not isinstance(plan['samples'], list) or any(not isinstance(s, dict) for s in plan['samples']):
        raise Stop('Sample objects required')
    if (stub != dict(kind='SYNTHETIC_TEST_ONLY', plan_sha256=plan_sha) or
            [s.get('image_id') for s in plan['samples']] != list(TARGETS)):
        raise Stop('Fixed four/plan-bound synthetic stub required')
    if set(plan['limits']) != set(LIMITS):
        raise Stop('Resource contract mismatch')
    for name, ceiling in LIMITS.items():
        value = plan['limits'][name]
        if type(value) is not int or value <= 0 or value > ceiling:
            raise Stop('Invalid resource limit')
    if plan['limits']['free_bytes'] != LIMITS['free_bytes']:
        raise Stop('Free-disk floor cannot be lowered')
    expected_keys = {'image_id', 'split', 'label', 'expected_shape', 'image_sha256',
                     'annotation_sha256', 'stac_sha256', 'license'}
    for sample in plan['samples']:
        if (set(sample) != expected_keys or type(sample['image_id']) is not int or
                sample['split'] != TARGETS[sample['image_id']] or
                sample['label'] not in ('zero', 'one', 'half') or
                sample['expected_shape'] != [4, 4] or
                any(type(v) is not int for v in sample['expected_shape']) or
                sample['license'] != 'CC-BY-4.0'):
            raise Stop('Synthetic sample split/shape/rights mismatch')
        for key in ('image_sha256', 'annotation_sha256', 'stac_sha256'):
            value = sample[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise Stop('Missing source/evidence pin')


def product_writer(budget):
    def write(path, array, profile):
        from rasterio.io import MemoryFile
        with MemoryFile() as memory:
            with memory.open(**profile) as dst:
                dst.write(array)
            data = memory.read()
        if len(data) > budget.limits['product_bytes']:
            raise Stop('Encoded product exceeds cap')
        budget.write(Path(path).relative_to(budget.root).as_posix(), data)
    return write


def deny_python_network(event, args):
    if event.startswith('socket.') or event in ('urllib.Request', 'subprocess.Popen', 'os.system'):
        raise Stop('Synthetic worker network/process call denied')


def _synthetic_run(plan, plan_sha, stub, output):
    """Private in-process test core. OS limits are enforced by run() around the worker."""
    import m1_5d_zenodo_prepare as converter
    validate_plan(plan, plan_sha, stub)
    budget = Budget.create(output, plan['limits'])
    result = dict(status='FAIL', mode='SYNTHETIC_ONLY', formal_r1='HOLD',
                  G3='PENDING', G4='PENDING', samples=[])
    try:
        for sample in plan['samples']:
            row, rgb, annotation, stac = fixture_assets(sample['image_id'], sample['label'])
            if [digest(x) for x in (rgb, annotation, stac)] != [
                    sample['image_sha256'], sample['annotation_sha256'], sample['stac_sha256']]:
                raise Stop('Synthetic source/evidence SHA mismatch')
            stem = row['oam_id']+'_'+str(row['image_id'])
            inputs = budget.root/'inputs'
            rgb_name = f'inputs/extracted/dataset/holdout/images/{stem}.tif'
            ann_name = f'inputs/extracted/dataset/holdout/masks/{stem}.png'
            stac_name = f'inputs/evidence/{row["oam_id"]}-stac.json'
            budget.write(rgb_name, rgb)
            budget.write(ann_name, annotation)
            budget.write(stac_name, stac, report=True)
            proof = dict(source_hashes={stem+'.tif': digest(rgb), stem+'.png': digest(annotation)},
                         stac_path=row['oam_id']+'-stac.json',
                         source_evidence=dict(snapshot_sha256=digest(stac), url='synthetic:test-only'),
                         review=dict(coverage='ALL_VALID', alignment='PASS', quality='PASS',
                                     black_wedge=False, basis='SYNTHETIC_TEST_ONLY fixture'))
            passes = []
            for name in ('primary', 'repeat'):
                converted = converter.convert(
                    inputs, row, proof, [1, 2], inputs/'evidence', budget.root/name,
                    product_writer=product_writer(budget), preview=False,
                    expected_shape=sample['expected_shape'])
                passes.append(converted)
            if passes[0]['output_hashes'] != passes[1]['output_hashes']:
                raise Stop('Repeat product SHA mismatch')
            result['samples'].append(dict(image_id=row['image_id'], split=sample['split'],
                                          primary=passes[0], repeat=passes[1]))
        if len(result['samples']) != 4 or budget.summary()['unresolved']:
            raise Stop('Incomplete batch')
        result['status'] = 'SYNTHETIC_PASS'
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result['budget_before_result'] = budget.summary()
        budget.write('result.json', encoded(result), report=True)
    return result


class WindowsJob:
    """Verified Job Object cap, active-process count one, kill-on-close."""

    def __init__(self, memory_bytes):
        if sys.platform != 'win32':
            raise Stop('Windows hard enforcement unavailable')
        class Basic(ctypes.Structure):
            _fields_ = [('user', ctypes.c_int64), ('job', ctypes.c_int64),
                        ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('processes', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ('reads', 'writes', 'other', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        self.info_type = Extended
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'IsProcessInJob': ([wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL)}
        for name, (args, result) in signatures.items():
            getattr(self.kernel, name).argtypes = args
            getattr(self.kernel, name).restype = result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise Stop('CreateJobObject failed')
        info = Extended()
        info.basic.flags = 0x2000 | 0x100 | 0x200 | 0x8
        info.basic.processes = 1
        info.process_memory = info.job_memory = memory_bytes
        try:
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise Stop('Hard memory/process limit installation failed')
            checked = self.query()
            if (checked.basic.flags & info.basic.flags != info.basic.flags or
                    checked.basic.processes != 1 or checked.process_memory != memory_bytes or
                    checked.job_memory != memory_bytes):
                raise Stop('Hard limit query mismatch')
        except BaseException:
            self.close()
            raise

    def query(self):
        info = self.info_type()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info), None):
            raise Stop('Job query failed')
        return info

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process._handle)):
            raise Stop('Worker job assignment failed')
        assigned = wintypes.BOOL()
        if (not self.kernel.IsProcessInJob(wintypes.HANDLE(process._handle), self.handle, ctypes.byref(assigned))
                or not assigned.value):
            raise Stop('Worker membership not verified')

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    @classmethod
    def require_current(cls, memory_bytes):
        probe = cls(memory_bytes)
        try:
            current = probe.info_type()
            if (not probe.kernel.QueryInformationJobObject(
                    None, 9, ctypes.byref(current), ctypes.sizeof(current), None) or
                    current.process_memory != memory_bytes or current.job_memory != memory_bytes or
                    current.basic.processes != 1 or current.basic.flags & 0x2308 != 0x2308):
                raise Stop('Worker requires the verified enclosing Job Object')
        finally:
            probe.close()


def limited_process(code, arguments, memory_bytes, deadline_seconds):
    """The trusted bootstrap waits for job assignment before importing worker code."""
    job = WindowsJob(memory_bytes)
    process = None
    started = time.monotonic()
    try:
        bootstrap = "import sys; sys.stdin.buffer.read(1); " + code
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', GDAL_NUM_THREADS='1',
                           GDAL_PAM_ENABLED='NO', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')
        # Windows venv python.exe is a redirector that can spawn the base Python.
        # Launch the interpreter itself so the one-process Job also covers startup.
        interpreter = getattr(sys, '_base_executable', sys.executable)
        process = subprocess.Popen([interpreter, '-I', '-B', '-S', '-c', bootstrap, *arguments],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, env=environment)
        job.assign(process)
        process.stdin.write(b'1')
        process.stdin.flush()
        process.stdin.close()
        try:
            remaining = deadline_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise Stop('Worker deadline exceeded')
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise Stop('Worker deadline exceeded') from exc
        peak = job.query().peak_job
        return dict(exit_code=exit_code, hard_job_memory_bytes=memory_bytes,
                    peak_job_commit_bytes=peak, active_process_limit=1,
                    job_membership_verified=True, elapsed_seconds=time.monotonic()-started)
    finally:
        job.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()


def run(plan_path, plan_sha, stub_path, stub_sha, output):
    plan = checked_json(plan_path, plan_sha)
    stub = checked_json(stub_path, stub_sha)
    validate_plan(plan, plan_sha, stub)
    ordinary(output)
    if Path(output).exists():
        raise Stop('Output must be new')
    # No callable real-data path exists until reliable native network isolation
    # and separately approved real input/review bindings have been implemented.
    code = ("sys.path[:0]=" + repr(sys.path) + "; import runpy; "
            "script=sys.argv.pop(1); runpy.run_path(script, run_name='__main__')")
    arguments = [str(Path(__file__).resolve()), '--worker', '--plan', str(Path(plan_path).resolve()),
                 '--plan-sha256', plan_sha, '--stub', str(Path(stub_path).resolve()),
                 '--stub-sha256', stub_sha, '--output', str(Path(output).absolute())]
    try:
        resource = limited_process(code, arguments, plan['limits']['memory_bytes'],
                                   plan['limits']['deadline_seconds'])
        success = resource['exit_code'] == 0
    except Stop as exc:
        resource = dict(error=str(exc))
        success = False
    if not Path(output).exists():
        raise Stop('Worker failed before output creation: '+str(resource))
    budget = Budget.retained(output, plan['limits'])
    result_path = Path(output)/'result.json'
    result = json.loads(result_path.read_bytes()) if result_path.exists() else {}
    complete = (result.get('status') == 'SYNTHETIC_PASS' and
                [s['image_id'] for s in result.get('samples', [])] == list(TARGETS) and
                not budget.summary()['unresolved'])
    final = dict(status='SYNTHETIC_PASS' if success and complete else 'FAIL',
                 plan_sha256=plan_sha, mode='SYNTHETIC_ONLY', formal_r1='HOLD',
                 G3='PENDING', G4='PENDING', resource=resource,
                 network='Python audit-denied; native OS isolation NOT_VERIFIED; real execution disabled',
                 budget_before_final=budget.summary())
    budget.write('execution.json', encoded(final), report=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--stub', type=Path, required=True, help='SYNTHETIC_TEST_ONLY stub')
    parser.add_argument('--stub-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            sys.addaudithook(deny_python_network)
            plan = checked_json(args.plan, args.plan_sha256)
            stub = checked_json(args.stub, args.stub_sha256)
            validate_plan(plan, args.plan_sha256, stub)
            WindowsJob.require_current(plan['limits']['memory_bytes'])
            _synthetic_run(plan, args.plan_sha256, stub, args.output)
            return 0
        result = run(args.plan, args.plan_sha256, args.stub, args.stub_sha256, args.output)
        print(json.dumps(result))
        return 0 if result['status'] == 'SYNTHETIC_PASS' else 1
    except Exception as exc:
        print(json.dumps(dict(status='FAIL', error_type=type(exc).__name__, error=str(exc))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
