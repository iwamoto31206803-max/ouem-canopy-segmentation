import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import research_checkpoint_audit as audit
import research_cuda as cuda
import research_val18_recovery as recovery
from test_research_cuda import ToyModel, synthetic_pool


def forbid(*_args, **_kwargs):
    pytest.fail('Unexpected CUDA or training call')


@pytest.fixture
def schedule(monkeypatch):
    fixture = cuda.make_schedule(synthetic_pool())
    monkeypatch.setattr(recovery, 'SCHEDULE', cuda.digest(fixture))
    return fixture


def test_schedule_checked_before_pilot_slice(schedule, monkeypatch):
    assert recovery.entries_for(schedule, True, cuda.digest) == schedule['val'][:1]
    assert recovery.entries_for(schedule, False, cuda.digest) == schedule['val']
    for changed in (schedule['val'][:1], list(reversed(schedule['val']))):
        invalid = dict(schedule, val=changed)
        monkeypatch.setattr(recovery, 'SCHEDULE', cuda.digest(invalid))
        with pytest.raises(ValueError, match='Fixed schedule'):
            recovery.entries_for(invalid, True, cuda.digest)


def test_cli_modes():
    args = ['--execute', '--recovery-id', 'fixture', '--threads', '1', '--timeout-seconds', '2',
            '--min-ram-bytes', str(8 * 1024**3), '--min-disk-bytes', str(16 * 1024**2)]
    for name in ('dataset', 'review', 'freeze', 'candidates', 'initializer', 'output'):
        args.extend(['--' + name, name])
    pins = ['--checkpoint', 'c', '--receipt', 'r', '--pins', 'p', '--pins-sha256', '0'*64]
    assert recovery.parse(args + ['--assess', '--step', '0']).step == 0
    assert recovery.parse(args + ['--pilot'] + pins).step == 280
    for invalid in (args + ['--pilot', '--step', '70'] + pins,
                    args + ['--assess', '--step', '1'],
                    args + ['--assess', '--step', '280'],
                    args[1:] + ['--assess', '--step', '0']):
        with pytest.raises(SystemExit):
            recovery.parse(invalid)


def test_frozen_source_file_content(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'REPO', tmp_path)
    names = ['scripts/research_cuda.py', 'scripts/research_data.py', 'scripts/m1_7_g3a_eval.py']
    (tmp_path / 'scripts').mkdir()
    for name in names:
        (tmp_path / name).write_bytes(b'pass\r\n')
    def git(*args):
        return {'status': b'', 'ls-tree': '\n'.join(names).encode(),
                'show': b'pass\n', 'rev-parse': b'tool\n'}[args[0]]
    monkeypatch.setattr(recovery, 'git', git)
    assert len(recovery.sources()['historical_python_sha256_lf']) == 3
    (tmp_path / names[1]).write_bytes(b'# changed\n')
    with pytest.raises(ValueError, match='drift'):
        recovery.sources()


def test_weight_load_verifies_full_bytes_first(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pth'
    torch.save({'model': torch.zeros(1)}, path)
    pin, size = recovery.sha(path), path.stat().st_size
    original = torch.load
    seen = []
    def load(stream, **kwargs):
        seen.append(kwargs)
        return original(stream, **kwargs)
    monkeypatch.setattr(torch, 'load', load)
    for wrong_sha, wrong_size in (('0'*64, size), (pin, size+1)):
        with pytest.raises(ValueError):
            recovery.verified_load(path, wrong_sha, wrong_size, torch)
    assert not seen
    assert recovery.verified_load(path, pin, size, torch)['model'].item() == 0
    assert seen == [{'map_location': 'cpu', 'weights_only': True}]


@pytest.fixture
def bindings(schedule):
    data = SimpleNamespace(identity={'fixture': 'v5'})
    pins = dict(sha256='1'*64)
    receipt = dict(checkpoint_sha256=pins['sha256'], segment_id='e'*32, active_seconds=2.0)
    binding = dict(run_id=recovery.RUN, mode='main', inputs=data.identity, commit=recovery.TRAINER,
                   freeze_sha256=cuda.FREEZE, initializer_sha256=cuda.INITIALIZER,
                   schedule_sha256=recovery.SCHEDULE, lineage_id='a'*32, environment={'satlas': '0.3.1'},
                   config=dict(seed=0, batch=1, precision='FP32', scheduler=None, adamw=cuda.ADAMW,
                               seconds=900.0, vram_bytes=12*1024**3))
    state = dict(kind=cuda.KIND, binding=binding, schedule=schedule, completed_steps=280,
                 next_item=None, model={}, optimizer={}, cpu_rng=None, cuda_rng=None, python_rng=None,
                 fingerprints={}, parent=None, segment_id=receipt['segment_id'], active_seconds=1.0,
                 backup_checkpoint=f'/old/{recovery.RUN}/step_280/checkpoint_280.pth')
    return data, pins, receipt, state


@pytest.mark.parametrize('mutation', ['none', 'step', 'mode', 'segment', 'config', 'lineage',
                                      'next_item', 'receipt_sha', 'freeze'])
def test_checkpoint_contract(bindings, schedule, mutation):
    data, pins, receipt, state = bindings
    if mutation == 'none':
        assert recovery.validate_state(state, receipt, pins, 280, data, schedule, cuda)[
            'independent_lineage'] == 'UNVERIFIED'
        return
    values = {
        'step': (state, 'completed_steps', 210),
        'mode': (state['binding'], 'mode', 'smoke'),
        'segment': (receipt, 'segment_id', 'f'*32),
        'config': (state['binding'], 'config', {}),
        'lineage': (pins, 'lineage_id', 'b'*32),
        'next_item': (state, 'next_item', {}),
        'receipt_sha': (receipt, 'checkpoint_sha256', 'f'*64),
        'freeze': (state['binding'], 'freeze_sha256', 'f'*64),
    }
    obj, key, value = values[mutation]
    obj[key] = value
    with pytest.raises(ValueError):
        recovery.validate_state(state, receipt, pins, 280, data, schedule, cuda)


@pytest.mark.parametrize('pilot', [True, False])
@pytest.mark.parametrize('failure', [None, 'freeze', 'ram', 'model_mutation'])
def test_worker_reuses_evaluator_and_panel(schedule, tmp_path, monkeypatch, pilot, failure):
    data = synthetic_pool()
    x = torch.zeros(1, 3, 224, 224)
    y = torch.zeros(1, 224, 224, dtype=torch.long)
    data.patch = lambda item, split: (x, y, {})
    data.identity = dict(fixture='v5')
    monkeypatch.setattr(cuda, 'IDENTITY', cuda.digest(data.identity))
    monkeypatch.setattr(cuda, 'ResearchData', lambda *_a, **_k: data)
    monkeypatch.setattr(cuda, 'read_manifest', lambda _: {'identity': {} if failure == 'freeze' else data.identity})
    monkeypatch.setattr(recovery, 'sha', lambda path: cuda.FREEZE if path == 'freeze' else cuda.INITIALIZER)
    monkeypatch.setattr(recovery, 'sources', lambda: {})
    monkeypatch.setattr(recovery, 'peak_memory', lambda: {'test': 1})
    monkeypatch.setattr(audit, 'available_memory', lambda: 0 if failure == 'ram' else 8 * 1024**3)
    monkeypatch.setattr(recovery.importlib.metadata, 'version', lambda _: '0.3.1')
    monkeypatch.setitem(sys.modules, 'torchvision', SimpleNamespace(__version__='fixture'))
    monkeypatch.setitem(sys.modules, 'satlaspretrain_models', SimpleNamespace())
    monkeypatch.setattr(torch.cuda, 'is_available', forbid)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbid)
    monkeypatch.setattr(cuda, 'run', forbid)
    monkeypatch.setattr(cuda, 'update', forbid)
    model = ToyModel()
    monkeypatch.setattr(recovery, 'model_for', lambda *_: (
        model, {'model_fingerprint': cuda.fingerprint(model.state_dict())}, {}))
    if failure == 'model_mutation':
        evaluate = cuda.evaluate
        def mutate(*args):
            metrics = evaluate(*args)
            with torch.no_grad():
                model.head.layers.bias.add_(1)
            return metrics
        monkeypatch.setattr(cuda, 'evaluate', mutate)
    args = SimpleNamespace(step=280, pilot=pilot, threads=1, min_ram_bytes=1, freeze='freeze',
                           initializer='initial', dataset=None, review=None, candidates=None,
                           recovery_id='fixture')
    if failure:
        with pytest.raises(SystemExit):
            recovery.worker(args, tmp_path, {})
        assert (tmp_path / 'failure.json').exists()
        assert not (tmp_path / 'metrics.json').exists()
        return
    recovery.worker(args, tmp_path, {})
    provenance = json.loads((tmp_path / 'provenance.json').read_bytes())
    entries = schedule['val'][:1] if pilot else schedule['val']
    assert provenance['full_val'] == schedule['val']
    assert provenance['non_acceptance'] is pilot
    metrics = json.loads((tmp_path / 'metrics.json').read_bytes())
    assert metrics['ce_valid_pixels'] == (50176 if pilot else 903168)
    for entry in entries:
        with Image.open(tmp_path / f"{entry['image_id']}.png") as img:
            assert img.size == (1120, 224)
    invalid = copy.deepcopy(metrics)
    invalid['windows'][0]['tn'] -= 1
    with pytest.raises(ValueError, match='totals'):
        recovery.check_metrics(invalid, entries)


@pytest.mark.parametrize('step', [0, 280])
def test_model_load_or_step0_constructor(step, bindings, schedule, tmp_path, monkeypatch):
    data, pins, receipt, state = bindings
    model = ToyModel()
    model.backbone = torch.nn.Linear(1, 1)
    model.fpn = torch.nn.Linear(1, 1)
    official = {'backbone': model.backbone.state_dict(), 'fpn': model.fpn.state_dict()}
    state['model'] = model.state_dict()
    state['fingerprints']['model'] = cuda.fingerprint(state['model'])
    monkeypatch.setitem(sys.modules, 'satlaspretrain_models.model',
                        SimpleNamespace(adjust_state_dict_prefix=lambda weights, name, *_: weights[name]))
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbid)
    monkeypatch.setattr(torch.optim, 'AdamW', forbid)
    strict = []
    load = model.load_state_dict
    def load_model(saved, *, strict=True):
        assert strict is True
        assert saved is state['model']
        return load(saved, strict=strict)
    monkeypatch.setattr(model, 'load_state_dict', lambda *a, **kw: (
        strict.append(kw['strict']), load_model(*a, **kw))[1])
    satlas = SimpleNamespace(Model=lambda **kw: model, Backbone=SimpleNamespace(SWINB=1),
                             Head=SimpleNamespace(SEGMENT=2))
    pins.update(step=280, size=100, receipt_sha256='2'*64, evidence_ref='review record')
    monkeypatch.setattr(recovery, 'STEP280', (100, '1'*64, '2'*64))
    args = SimpleNamespace(step=step, initializer='initializer', pins='pins', pins_sha256='3'*64,
                           receipt='receipt', checkpoint='checkpoint')
    monkeypatch.setattr(recovery, 'verified_load', lambda path, *_: official if path == 'initializer' else state)
    monkeypatch.setattr(audit, 'pinned_json', lambda path, _: pins if path == 'pins' else receipt)
    result, provenance, _ = recovery.model_for(args, data, schedule, cuda, torch, satlas)
    assert result is model and strict == ([True] if step else [])
    assert provenance.get('original_head_equality') == ('UNVERIFIED' if step == 0 else None)
    if step == 0:
        assert set(provenance['official_components']) == {'backbone', 'fpn'}


def test_private_destinations_and_copy_receipt_last(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    copy_path = tmp_path / 'copy'
    names = {'metrics.json', 'provenance.json', '1241.png'}
    for name in names:
        (output / name).write_bytes(name.encode())
    with pytest.raises(ValueError):
        recovery.protected_destination(output / 'inside', [output])
    with pytest.raises(ValueError):
        recovery.protected_destination(output, [])
    order = []
    write = recovery.write_json
    def record(path, value):
        order.append(path.name)
        return write(path, value)
    monkeypatch.setattr(recovery, 'write_json', record)
    recovery.publish(output, copy_path, names)
    assert order == ['sha256.json', 'completion.json', 'completion.json']
    receipt = json.loads((copy_path / 'completion.json').read_bytes())
    assert receipt['local_copy'] == 'LOCAL_COPY_READBACK_PASS'
    assert receipt['cloud'] == 'CLOUD_DURABILITY_UNVERIFIED'
    assert receipt['index_sha256'] == recovery.sha(copy_path / 'sha256.json')


def test_partial_copy_has_no_completion_receipt(tmp_path, monkeypatch):
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'metrics.json').write_bytes(b'{}')
    copy_path = tmp_path / 'copy'
    def interrupt(src, dest):
        dest.write(b'{')
        raise InterruptedError()
    monkeypatch.setattr(recovery.shutil, 'copyfileobj', interrupt)
    with pytest.raises(InterruptedError):
        recovery.publish(output, copy_path, {'metrics.json'})
    assert not (output / 'completion.json').exists()
    assert not (copy_path / 'completion.json').exists()


@pytest.mark.parametrize('problem', ['wrong_hash', 'malformed', 'missing'])
def test_receipt_gate_before_checkpoint_load(bindings, schedule, tmp_path, monkeypatch, problem):
    model_package = SimpleNamespace(adjust_state_dict_prefix=forbid)
    monkeypatch.setitem(sys.modules, 'satlaspretrain_models.model', model_package)
    _, pins, _, _ = bindings
    receipt = tmp_path / 'receipt.json'
    if problem != 'missing':
        receipt.write_bytes(b'invalid json')
    claims = dict(pins, step=280, size=100, evidence_ref='reviewed',
                  receipt_sha256=recovery.sha(receipt) if problem == 'malformed' else '2'*64)
    contract = tmp_path / 'pins.json'
    recovery.write_json(contract, claims)
    monkeypatch.setattr(recovery, 'STEP280', (100, '1'*64, claims['receipt_sha256']))
    monkeypatch.setattr(recovery, 'verified_load', forbid)
    args = SimpleNamespace(step=280, pins=contract, pins_sha256=recovery.sha(contract), receipt=receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.model_for(args, None, schedule, cuda, torch, None)


@pytest.mark.parametrize('failure', ['disk', 'timeout', 'interrupt'])
def test_watchdog_clears_child_and_leaves_no_receipt(tmp_path, monkeypatch, failure):
    inputs, reports = tmp_path / 'inputs', tmp_path / 'reports'
    inputs.mkdir()
    reports.mkdir()
    dataset = inputs / 'dataset'
    dataset.mkdir()
    artifact = inputs / 'fixture'
    artifact.touch()
    output = reports / 'fresh'
    args = SimpleNamespace(dataset=dataset, review=artifact, freeze=artifact, candidates=artifact,
                           initializer=artifact, checkpoint=None, receipt=None, pins=None,
                           output=output, local_copy=None, min_disk_bytes=1, timeout_seconds=3)
    monkeypatch.setattr(recovery, 'sources', lambda: {})
    monkeypatch.setattr(recovery.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=0 if failure == 'disk' else 10))
    calls = []
    class Process:
        def start(self):
            calls.append('start')
        def join(self, timeout=None):
            calls.append(timeout)
            if failure == 'interrupt' and timeout:
                raise KeyboardInterrupt()
        def is_alive(self):
            return True
        def kill(self):
            calls.append('kill')
        def close(self):
            calls.append('close')
    monkeypatch.setattr(recovery.multiprocessing, 'get_context',
                        lambda _: SimpleNamespace(Process=lambda **_: Process()))
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else ValueError):
        recovery.execute(args)
    assert calls == ([] if failure == 'disk' else ['start', 3, 'kill', None, 'close'])
    assert not (output / 'completion.json').exists()


def test_symlink_parent_alias_cannot_write_into_inputs(tmp_path):
    original = tmp_path / 'input'
    original.mkdir()
    alias = tmp_path / 'linked'
    try:
        alias.symlink_to(original, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation unavailable on this Windows host')
    with pytest.raises(ValueError, match='outside input'):
        recovery.protected_destination(alias / 'new', [original.resolve()])
