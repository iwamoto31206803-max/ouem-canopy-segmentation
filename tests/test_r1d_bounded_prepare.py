"""Synthetic-only fixed-four safety and real Windows resource enforcement."""
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import m1_5d_zenodo_prepare as converter
import r1d_bounded_prepare as bounded


@pytest.fixture
def contract():
    plan = bounded.synthetic_plan()
    plan_sha = bounded.digest(bounded.encoded(plan))
    stub = dict(kind='SYNTHETIC_TEST_ONLY', plan_sha256=plan_sha)
    return plan, plan_sha, stub


def seal(plan):
    digest = bounded.digest(bounded.encoded(plan))
    return digest, dict(kind='SYNTHETIC_TEST_ONLY', plan_sha256=digest)


def inputs(tmp_path, contract):
    plan, digest, stub = contract
    plan_path = tmp_path/'plan.json'
    stub_path = tmp_path/'stub.json'
    plan_path.write_bytes(bounded.encoded(plan))
    stub_path.write_bytes(bounded.encoded(stub))
    return plan_path, digest, stub_path, bounded.digest(stub_path.read_bytes())


def test_four_success_deterministic_readback_uniform_and_no_network(contract, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network must never be called')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    plan, digest, stub = contract
    result = bounded._synthetic_run(plan, digest, stub, tmp_path/'run')
    assert result['status'] == 'SYNTHETIC_PASS'
    assert [s['image_id'] for s in result['samples']] == list(bounded.TARGETS)
    assert result['formal_r1'] == 'HOLD'
    assert result['G3'] == result['G4'] == 'PENDING'
    assert len(list((tmp_path/'run').rglob('*.tif'))) == 36  # 32 products +4 synthetic sources
    assert not list((tmp_path/'run'/'primary').rglob('preview.png'))
    for sample in result['samples']:
        assert sample['primary']['output_hashes'] == sample['repeat']['output_hashes']
        primary = tmp_path/'run'/'primary'/sample['split']/str(sample['image_id'])
        with rasterio.open(primary/'canopy_mask.tif') as src:
            mask = src.read()
        with rasterio.open(primary/'canopy_fraction.tif') as src:
            fraction = src.read()
        with rasterio.open(primary/'valid_fraction.tif') as src:
            assert np.all(src.read() == 1)
        assert np.array_equal(mask, (fraction >= .5).astype('uint8'))
        if sample['image_id'] == 717:
            assert np.all(mask == 0)
        if sample['image_id'] == 1054:
            assert np.all(mask == 1)
    recovered = bounded.Budget.retained(tmp_path/'run', plan['limits'])
    summary = recovered.summary()
    assert not summary['unresolved']
    assert summary['retained_file_bytes'] <= summary['charged_upper_bound'] <= plan['limits']['write_bytes']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(source_generation='hf'),
    lambda p: p.update(schema='wrong'),
    lambda p: p['samples'].pop(),
    lambda p: p['samples'].append(copy.deepcopy(p['samples'][0])),
    lambda p: p['samples'][0].update(image_id=9),
    lambda p: p['samples'][0].update(image_id=717.0),
    lambda p: p['samples'][0].update(split='val'),
    lambda p: p['samples'][0].update(expected_shape=[5, 4]),
    lambda p: p['samples'][0].update(expected_shape=[4.0, 4.0]),
    lambda p: p['samples'][0].update(license='CC-BY-NC-4.0'),
    lambda p: p['samples'][0].update(path='C:/real/source.tif'),
    lambda p: p['samples'][0].pop('image_sha256'),
    lambda p: p['limits'].update(write_bytes=64*1024**2+1),
    lambda p: p['limits'].update(free_bytes=1),
])
def test_plan_rejections_before_output(contract, tmp_path, mutation):
    plan = copy.deepcopy(contract[0])
    mutation(plan)
    digest, stub = seal(plan)
    with pytest.raises((bounded.Stop, KeyError)):
        bounded._synthetic_run(plan, digest, stub, tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_no_stub_or_wrong_binding(contract, tmp_path):
    plan, digest, stub = contract
    for bad in ({}, dict(stub, plan_sha256='0'*64), dict(stub, approved=True)):
        with pytest.raises(bounded.Stop):
            bounded._synthetic_run(plan, digest, bad, tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_wrong_source_or_stac_hash_retains_failure(contract, tmp_path):
    for key in ['image_sha256', 'annotation_sha256', 'stac_sha256']:
        plan = copy.deepcopy(contract[0])
        plan['samples'][0][key] = '0'*64
        digest, stub = seal(plan)
        output = tmp_path/key
        with pytest.raises(bounded.Stop, match='SHA mismatch'):
            bounded._synthetic_run(plan, digest, stub, output)
        result = json.loads((output/'result.json').read_bytes())
        assert result['status'] == 'FAIL' and result['samples'] == []
        assert not list(output.rglob('*.tif'))


def test_real_mode_rejected_without_opening_source(contract, tmp_path, monkeypatch):
    plan = copy.deepcopy(contract[0])
    plan['mode'] = 'REAL'
    digest, stub = seal(plan)
    def forbidden(*args, **kwargs):
        raise AssertionError('Source access')
    monkeypatch.setattr(bounded, 'fixture_assets', forbidden)
    with pytest.raises(bounded.Stop, match='REAL_EXECUTION_DISABLED'):
        bounded._synthetic_run(plan, digest, stub, tmp_path/'out')
    paths = inputs(tmp_path, (plan, digest, stub))
    with pytest.raises(bounded.Stop, match='REAL_EXECUTION_DISABLED'):
        bounded.run(*paths, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_plan_file_hash_mismatch(contract, tmp_path):
    paths = list(inputs(tmp_path, contract))
    paths[1] = '0'*64
    with pytest.raises(bounded.Stop, match='file SHA'):
        bounded.run(*paths, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('worker', [False, True])
def test_real_cli_fails_closed(contract, tmp_path, worker):
    plan = copy.deepcopy(contract[0])
    plan['mode'] = 'REAL'
    digest, stub = seal(plan)
    paths = inputs(tmp_path, (plan, digest, stub))
    command = [sys.executable, '-B', bounded.__file__, '--plan', str(paths[0]),
               '--plan-sha256', paths[1], '--stub', str(paths[2]),
               '--stub-sha256', paths[3], '--output', str(tmp_path/'out')]
    if worker:
        command.append('--worker')
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert 'REAL_EXECUTION_DISABLED' in result.stdout
    assert not (tmp_path/'out').exists()


def test_control_size_and_duplicate_json_keys(tmp_path):
    path = tmp_path/'plan.json'
    path.write_bytes(b'x'*65537)
    with pytest.raises(bounded.Stop, match='64 KiB'):
        bounded.checked_json(path, bounded.digest(path.read_bytes()))
    path.write_bytes(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match='Duplicate'):
        bounded.checked_json(path, bounded.digest(path.read_bytes()))


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a//b', 'a\\b', 'C:/a', 'a/NUL', 'x.'])
def test_path_escape_rejected(tmp_path, name):
    budget = bounded.Budget.create(tmp_path/'out', dict(bounded.LIMITS))
    with pytest.raises(bounded.Stop):
        budget.write(name, b'x')
    assert budget.charged == 0


def test_preexisting_protected_and_reparse_output(tmp_path, monkeypatch):
    root = tmp_path/'exists'
    root.mkdir()
    with pytest.raises(bounded.Stop):
        bounded.Budget.create(root, dict(bounded.LIMITS))
    monkeypatch.setattr(bounded, 'PROTECTED', [tmp_path])
    with pytest.raises(bounded.Stop, match='Protected'):
        bounded.Budget.create(tmp_path/'new', dict(bounded.LIMITS))
    monkeypatch.setattr(bounded, 'PROTECTED', [])
    budget = bounded.Budget.create(tmp_path/'budget', dict(bounded.LIMITS))
    (budget.root/'file').write_bytes(b'old')
    with pytest.raises(bounded.Stop):
        budget.write('file', b'new')
    assert (budget.root/'file').read_bytes() == b'old'
    original = bounded.ordinary
    def reparse(path):
        if Path(path).name == 'linked':
            raise bounded.Stop('Symlink/reparse point forbidden')
        original(path)
    monkeypatch.setattr(bounded, 'ordinary', reparse)
    with pytest.raises(bounded.Stop):
        budget.write('linked', b'new')


def test_budget_duplicate_partial_and_recovery(tmp_path, monkeypatch):
    budget = bounded.Budget.create(tmp_path/'out', dict(bounded.LIMITS))
    budget.write('first', b'ab')
    with pytest.raises(bounded.Stop):
        budget.write('FIRST', b'no')
    def interrupted(stream, data):
        stream.write(data[:2])
        stream.flush()
        raise OSError('injected partial failure')
    monkeypatch.setattr(budget, '_payload', interrupted)
    with pytest.raises(OSError):
        budget.write('second', b'abcdef')
    recovered = bounded.Budget.retained(budget.root, budget.limits)
    assert recovered.charged == budget.charged
    assert recovered.summary()['unresolved'] == ['second']
    assert (budget.root/'second.partial').read_bytes() == b'ab'
    assert not (budget.root/'second').exists()
    with pytest.raises(bounded.Stop):
        recovered.write('second', b'abcdef')


def test_partial_hash_and_committed_hash_changes_stop(tmp_path, monkeypatch):
    budget = bounded.Budget.create(tmp_path/'out', dict(bounded.LIMITS))
    def corrupt(stream, data):
        stream.write(b'x'*len(data))
    monkeypatch.setattr(budget, '_payload', corrupt)
    with pytest.raises(bounded.Stop, match='Partial changed'):
        budget.write('changed', b'abc')
    assert not (budget.root/'changed').exists()
    assert (budget.root/'changed.partial').exists()
    another = bounded.Budget.create(tmp_path/'committed', dict(bounded.LIMITS))
    another.write('good', b'abc')
    (another.root/'good').write_bytes(b'bad')
    with pytest.raises(bounded.Stop, match='Committed file changed'):
        bounded.Budget.retained(another.root, another.limits)


def test_corrupt_retained_journal_stops(tmp_path):
    budget = bounded.Budget.create(tmp_path/'out', dict(bounded.LIMITS))
    with (budget.root/'journal.jsonl').open('ab') as stream:
        stream.write(b'truncated')
    with pytest.raises(bounded.Stop, match='reconciliation'):
        bounded.Budget.retained(budget.root, budget.limits)


def test_cumulative_payload_and_control_report_limits(tmp_path):
    limits = dict(bounded.LIMITS, write_bytes=2100)
    budget = bounded.Budget.create(tmp_path/'out', limits)
    budget.write('primary', b'x'*20)
    budget.write('repeat', b'x'*20)
    before = budget.charged
    with pytest.raises(bounded.Stop):
        budget.write('report.json', b'{}', report=True)
    assert budget.charged == before
    budget = bounded.Budget.create(tmp_path/'reports', dict(bounded.LIMITS, report_bytes=1025))
    with pytest.raises(bounded.Stop):
        budget.write('report.json', b'{}', report=True)
    assert not (budget.root/'report.json').exists()


def test_free_space_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(bounded.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': 1})())
    with pytest.raises(bounded.Stop, match='free disk'):
        bounded.Budget.create(tmp_path/'out', dict(bounded.LIMITS))
    assert not (tmp_path/'out').exists()


def test_encoder_cap_and_injected_encode_failure(contract, tmp_path, monkeypatch):
    plan = copy.deepcopy(contract[0])
    plan['limits']['product_bytes'] = 1
    digest, stub = seal(plan)
    with pytest.raises(bounded.Stop, match='Encoded product'):
        bounded._synthetic_run(plan, digest, stub, tmp_path/'cap')
    assert not list((tmp_path/'cap'/'primary').rglob('*.tif'))
    assert json.loads((tmp_path/'cap'/'result.json').read_bytes())['status'] == 'FAIL'
    def fail(*args):
        def writer(*args):
            raise OSError('encoder failed')
        return writer
    monkeypatch.setattr(bounded, 'product_writer', fail)
    with pytest.raises(OSError):
        bounded._synthetic_run(*contract, tmp_path/'encoder')
    assert json.loads((tmp_path/'encoder'/'result.json').read_bytes())['samples'] == []


def test_mid_batch_fail_fast_and_existing_records_preserved(contract, tmp_path, monkeypatch):
    original = converter.convert
    calls = []
    def fail_on_second_image(root, row, *args, **kwargs):
        calls.append(row['image_id'])
        if row['image_id'] == 1054:
            raise ValueError('injected mid-batch error')
        return original(root, row, *args, **kwargs)
    monkeypatch.setattr(converter, 'convert', fail_on_second_image)
    with pytest.raises(ValueError):
        bounded._synthetic_run(*contract, tmp_path/'out')
    result = json.loads((tmp_path/'out'/'result.json').read_bytes())
    assert result['status'] == 'FAIL'
    assert [s['image_id'] for s in result['samples']] == [717]
    assert calls == [717, 717, 1054]
    assert len(list((tmp_path/'out'/'primary'/'train'/'717').glob('*.tif'))) == 4
    with pytest.raises(bounded.Stop):
        bounded._synthetic_run(*contract, tmp_path/'out')


def test_actual_converter_shape_gate(contract, tmp_path, monkeypatch):
    original = converter.output_grid
    def wrong_shape(*args):
        shape, transform, gsd = original(*args)
        return (shape[0]+1, shape[1]), transform, gsd
    monkeypatch.setattr(converter, 'output_grid', wrong_shape)
    with pytest.raises(ValueError, match='shape mismatch'):
        bounded._synthetic_run(*contract, tmp_path/'out')
    assert not list((tmp_path/'out'/'primary').rglob('*.tif'))


def test_repeat_mismatch_stops(contract, tmp_path, monkeypatch):
    original = converter.convert
    def mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        if Path(args[5]).name == 'repeat':
            result['output_hashes']['rgb.tif'] = '0'*64
        return result
    monkeypatch.setattr(converter, 'convert', mismatch)
    with pytest.raises(bounded.Stop, match='Repeat product SHA'):
        bounded._synthetic_run(*contract, tmp_path/'out')
    result = json.loads((tmp_path/'out'/'result.json').read_bytes())
    assert result['status'] == 'FAIL' and result['samples'] == []


def test_invalid_validity_token_never_translated(contract, tmp_path, monkeypatch):
    original = converter.validity
    def invalid(rgb, review):
        return original(rgb, dict(review, coverage='REVIEW_PASS'))
    monkeypatch.setattr(converter, 'validity', invalid)
    with pytest.raises(ValueError, match='VALIDITY'):
        bounded._synthetic_run(*contract, tmp_path/'out')


def test_native_platform_fail_closed(monkeypatch):
    monkeypatch.setattr(bounded.sys, 'platform', 'unsupported')
    with pytest.raises(bounded.Stop, match='unavailable'):
        bounded.WindowsJob(64*1024**2)


def test_failed_hard_limit_setup_prevents_output(contract, tmp_path, monkeypatch):
    def unavailable(*args):
        raise bounded.Stop('Hard enforcement installation failed')
    monkeypatch.setattr(bounded, 'WindowsJob', unavailable)
    with pytest.raises(bounded.Stop, match='Worker failed before output'):
        bounded.run(*inputs(tmp_path, contract), tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_python_network_audit_policy():
    for event in ['socket.__new__', 'socket.connect', 'socket.getaddrinfo',
                  'urllib.Request', 'subprocess.Popen', 'os.system']:
        with pytest.raises(bounded.Stop):
            bounded.deny_python_network(event, ())
    bounded.deny_python_network('open', ())


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job Object test')
def test_windows_memory_limit_is_enforced():
    code = ("import sys\nblocks=[]\ntry:\n"
            " while True: blocks.append(bytearray(8*1024**2))\n"
            "except MemoryError: sys.exit(37)\n")
    result = bounded.limited_process(code, [], 64*1024**2, 10)
    assert result['job_membership_verified']
    assert result['exit_code'] == 37
    assert 0 < result['peak_job_commit_bytes'] <= 64*1024**2


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job Object test')
def test_windows_deadline_kills_worker():
    started = time.monotonic()
    with pytest.raises(bounded.Stop, match='deadline'):
        bounded.limited_process('import time; time.sleep(30)', [], 64*1024**2, .2)
    assert time.monotonic()-started < 5


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job Object test')
def test_windows_process_tree_is_limited():
    code = ("import subprocess,sys\ntry:\n"
            " child=subprocess.run([sys.executable,'-S','-c','pass'])\n"
            " sys.exit(37 if child.returncode else 0)\n"
            "except OSError: sys.exit(37)\n")
    result = bounded.limited_process(code, [], 64*1024**2, 10)
    assert result['exit_code'] == 37
    assert result['active_process_limit'] == 1


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job Object test')
def test_network_call_is_audit_blocked_in_worker():
    code = ("sys.path[:0]=" + repr(sys.path) + "\n"
            "from r1d_bounded_prepare import deny_python_network, Stop\n"
            "import socket\nsys.addaudithook(deny_python_network)\n"
            "try: socket.socket()\nexcept Stop: sys.exit(37)\n")
    result = bounded.limited_process(code, [], 2*1024**3, 30)
    assert result['exit_code'] == 37


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows Job Object test')
def test_isolated_cli_worker_success(contract, tmp_path):
    result = bounded.run(*inputs(tmp_path, contract), tmp_path/'isolated')
    assert result['status'] == 'SYNTHETIC_PASS'
    assert result['resource']['hard_job_memory_bytes'] == 2*1024**3
    assert result['resource']['job_membership_verified']
    assert result['formal_r1'] == 'HOLD'
    persisted = json.loads((tmp_path/'isolated'/'execution.json').read_bytes())
    assert persisted == result
