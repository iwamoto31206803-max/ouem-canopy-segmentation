import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import research_checkpoint_audit as audit


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    inputs = {'fixture': 'v5'}
    schedule = dict(train=[{'step': i} for i in range(1, 281)],
                    val=[{'image_id': i} for i in audit.VAL])
    monkeypatch.setattr(audit, 'IDENTITY', audit.digest(inputs))
    monkeypatch.setattr(audit, 'SCHEDULE', audit.digest(schedule))
    monkeypatch.setattr(audit, 'available_memory', lambda: 32 * 1024**3)
    lineage = 'd' * 32
    expected = dict(lineage_id=lineage, seconds=900, vram_bytes=12 * 1024**3)
    env = dict(python='3.12', torch='test', torchvision='test', cuda='test', cudnn=1,
               satlas='0.3.1', gpu='test', total_vram=16 * 1024**3, devices=1)
    weight = torch.nn.Parameter(torch.ones(2))
    bias = torch.nn.Parameter(torch.ones(1))
    model = torch.nn.ParameterDict(dict(weight=weight, bias=bias))
    optimizer = torch.optim.AdamW(model.parameters(), **audit.ADAMW)
    (weight.sum() + bias.sum()).backward()
    optimizer.step()
    state = dict(kind=audit.KIND, completed_steps=280, next_item=None, schedule=schedule,
                 binding=dict(run_id=audit.RUN, lineage_id=lineage, mode='main', inputs=inputs,
                              freeze_sha256=audit.FREEZE, initializer_sha256=audit.INITIALIZER,
                              schedule_sha256=audit.SCHEDULE, commit=audit.COMMIT, environment=env,
                              config=dict(seed=0, batch=1, precision='FP32', scheduler=None,
                                          adamw=dict(audit.ADAMW, betas=list(audit.ADAMW['betas'])),
                                          seconds=900.0, vram_bytes=expected['vram_bytes'])),
                 backup_checkpoint=f'/historical/{audit.RUN}/step_280/checkpoint_280.pth',
                 segment_id=audit.SEGMENT, parent=None, active_seconds=310,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 python_rng=random.Random(0).getstate(),
                 cpu_rng=torch.Generator().manual_seed(0).get_state(),
                 cuda_rng=[torch.arange(16, dtype=torch.uint8)])
    state['fingerprints'] = {key: audit.fingerprint(state[field]) for key, field in audit.FIELDS.items()}

    def prepare(receipt_changes=None):
        ckpt, receipt, contract, output = (tmp_path / name for name in
                                           ('checkpoint.pth', 'runtime.json', 'expected.json', 'out.json'))
        torch.save(state, ckpt)
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        size = ckpt.stat().st_size
        monkeypatch.setattr(audit, 'SHA', sha)
        monkeypatch.setattr(audit, 'SIZE', size)
        values = dict(checkpoint_sha256=sha, segment_id=audit.SEGMENT, active_seconds=audit.ELAPSED)
        values.update(receipt_changes or {})
        receipt.write_text(json.dumps(values))
        contract.write_text(json.dumps(expected))
        return argparse.Namespace(checkpoint=ckpt, receipt=receipt, contract=contract, output=output,
                                  sha256=sha, size=size,
                                  receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                  contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest())

    return state, expected, prepare


def test_pass_is_cpu_only_and_explicit_about_missing_historical_evidence(checkpoint, monkeypatch):
    _, _, prepare = checkpoint
    args = prepare()
    before = random.getstate(), torch.get_rng_state()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA access'))
    assert audit.audit(args)
    result = json.loads(args.output.read_text())
    assert result['status'] == 'CPU_SERIALIZED_STATE_VERIFIED'
    assert result['historical_environment_status'] == 'HISTORICAL_ENVIRONMENT_UNVERIFIED'
    assert result['gpu_restore'] == 'NOT_RUN'
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256


@pytest.mark.parametrize('field,value', [
    ('run_id', 'other'), ('mode', 'smoke'), ('lineage_id', 'e'*32),
    ('schedule_sha256', '0'*64), ('commit', 'other'), ('config', {}),
])
def test_reject_wrong_binding(checkpoint, field, value):
    state, _, prepare = checkpoint
    state['binding'][field] = value
    assert not audit.audit(prepare())


@pytest.mark.parametrize('field,value', [
    ('completed_steps', 210), ('segment_id', 'e'*32), ('next_item', {}),
    ('schedule', {'train': [], 'val': []}), ('model', {'weight': torch.tensor(float('nan'))}),
    ('cpu_rng', torch.ones(16)), ('cuda_rng', []), ('python_rng', ('invalid',)),
])
def test_reject_wrong_state(checkpoint, field, value):
    state, _, prepare = checkpoint
    state[field] = value
    assert not audit.audit(prepare())


@pytest.mark.parametrize('field', audit.FIELDS)
@pytest.mark.parametrize('missing', (False, True))
def test_reject_bad_fingerprint(checkpoint, field, missing):
    state, _, prepare = checkpoint
    if missing:
        del state['fingerprints'][field]
    else:
        state['fingerprints'][field] = '0'*64
    assert not audit.audit(prepare())


@pytest.mark.parametrize('change', ('group', 'moment', 'empty', 'missing_key', 'extra_key', 'orphan_slot'))
def test_reject_optimizer_even_with_matching_fingerprint(checkpoint, change):
    state, _, prepare = checkpoint
    optimizer = state['optimizer']
    if change == 'group':
        optimizer['param_groups'][0]['lr'] = .1
    elif change == 'moment':
        optimizer['state'][0]['exp_avg_sq'] = torch.ones(3)
    elif change == 'empty':
        optimizer['state'].clear()
    elif change == 'missing_key':
        del optimizer['param_groups'][0]['foreach']
    elif change == 'extra_key':
        optimizer['param_groups'][0]['unknown'] = False
    else:
        optimizer['state'][2] = optimizer['state'].pop(0)
    state['fingerprints']['optimizer'] = audit.fingerprint(optimizer)
    assert not audit.audit(prepare())


@pytest.mark.parametrize('failure', ('size', 'sha', 'receipt_missing', 'receipt_malformed', 'ram'))
def test_guards_reject_before_load(checkpoint, monkeypatch, failure):
    _, _, prepare = checkpoint
    args = prepare()
    if failure == 'size':
        with args.checkpoint.open('ab') as stream:
            stream.write(b'x')
    elif failure == 'sha':
        with args.checkpoint.open('r+b') as stream:
            stream.write(b'x')
    elif failure == 'receipt_missing':
        args.receipt.unlink()
    elif failure == 'receipt_malformed':
        args.receipt.write_text('{')
        args.receipt_sha256 = hashlib.sha256(args.receipt.read_bytes()).hexdigest()
    else:
        monkeypatch.setattr(audit, 'available_memory', lambda: 1)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Unsafe load'))
    if failure == 'receipt_missing':
        with pytest.raises(ValueError):
            audit.audit(args)
    else:
        assert not audit.audit(args)


def test_bad_receipt_and_expected_environment(checkpoint):
    state, expected, prepare = checkpoint
    args = prepare({'active_seconds': 309})
    assert not audit.audit(args)
    args.output.unlink()
    expected['environment'] = dict(state['binding']['environment'], gpu='other')
    assert not audit.audit(prepare())
    args.output.unlink()
    expected['environment'] = state['binding']['environment'].copy()
    assert audit.audit(prepare())
    assert json.loads(args.output.read_text())['historical_provenance']['environment'] == 'PASS'


def test_output_must_be_fresh_and_distinct(checkpoint):
    _, _, prepare = checkpoint
    args = prepare()
    args.output = args.receipt
    with pytest.raises(FileExistsError):
        audit.audit(args)
    args.output = args.checkpoint
    with pytest.raises(ValueError):
        audit.audit(args)


def test_missing_independent_lineage_is_unverified(checkpoint, monkeypatch):
    state, expected, prepare = checkpoint
    del expected['lineage_id']
    args = prepare()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA access'))
    before = random.getstate(), torch.get_rng_state()
    assert audit.audit(args)
    report = json.loads(args.output.read_text())
    assert report['status'] == 'CPU_PAYLOAD_VERIFIED_LINEAGE_UNVERIFIED'
    assert report['gates']['independent_lineage'] == 'UNVERIFIED'
    assert report['lineage_source'] == 'CHECKPOINT_DERIVED'
    assert report['lineage_id'] == state['binding']['lineage_id']
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256


def test_sparse_adamw_slots_are_reported(checkpoint):
    state, _, prepare = checkpoint
    optimizer = state['optimizer']
    del optimizer['state'][0]
    state['fingerprints']['optimizer'] = audit.fingerprint(optimizer)
    args = prepare()
    assert audit.audit(args)
    assert json.loads(args.output.read_text())['optimizer_slots'] == {
        'parameters': 2, 'populated_slots': 1, 'absent_slots': 1}


def test_report_inside_repo_is_rejected(checkpoint):
    _, _, prepare = checkpoint
    args = prepare()
    args.output = Path(audit.__file__).resolve().parents[1] / args.output.name
    with pytest.raises(ValueError, match='outside the Git checkout'):
        audit.audit(args)
