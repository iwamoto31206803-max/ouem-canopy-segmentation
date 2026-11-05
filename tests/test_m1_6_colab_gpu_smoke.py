"""Offline contracts for the opt-in CUDA CLI; no real Satlas model is constructed."""
import copy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import m1_6_colab_gpu_smoke as gpu
from m1_4_bounded_overfit import fingerprint


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('Network forbidden'))


def arguments(tmp_path, *extra):
    return ['--data', str(tmp_path / 'data'), '--checkpoint', str(tmp_path / 'official.pth'),
            '--output', str(tmp_path / 'result'), *extra]


@pytest.mark.parametrize('extra', [['--max-steps', '0'], ['--max-steps', '21'], ['--max-steps', '1.5']])
def test_cli_step_bound(tmp_path, extra):
    with pytest.raises(SystemExit):
        gpu.parser().parse_args(arguments(tmp_path, *extra))


@pytest.mark.parametrize('extra', [['--restore-only'], ['--resume', 'old.pth'], ['--resume-sha256', '0'*64]])
def test_resume_arguments(tmp_path, extra):
    with pytest.raises(ValueError):
        gpu.reserve_output(gpu.parser().parse_args(arguments(tmp_path, *extra)))
    assert not (tmp_path / 'result').exists()


def test_fresh_external_output(tmp_path):
    args = gpu.parser().parse_args(arguments(tmp_path))
    gpu.reserve_output(args)
    with pytest.raises(FileExistsError):
        gpu.reserve_output(args)
    for output in [args.data, args.data / 'new', gpu.REPO / 'new-run']:
        args.output = output
        with pytest.raises(ValueError):
            gpu.reserve_output(args)


def test_cuda_missing_records_not_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert gpu.main(arguments(tmp_path)) == 2
    record = json.loads((tmp_path / 'result/result.json').read_text())
    assert record['status'] == 'CUDA_NOT_VERIFIED'
    assert record['stage'] == 'environment' and not record['cuda_available']
    assert 'steps' not in record and record['total_seconds'] >= 0


def test_failure_records_stage_and_stops(tmp_path, monkeypatch):
    def failure(args, record):
        record['stage'] = 'optimizer_step'
        raise RuntimeError('simulated OOM')
    monkeypatch.setattr(gpu, 'run', failure)
    assert gpu.main(arguments(tmp_path)) == 1
    record = json.loads((tmp_path / 'result/result.json').read_text())
    assert record['status'] == 'FAIL_STOPPED' and record['stage'] == 'optimizer_step'
    assert record['exception']['message'] == 'simulated OOM'
    assert not (tmp_path / 'result/gpu_smoke.pth').exists()


def test_safe_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   cpu_rng=torch.get_rng_state(), cuda_rng=[], completed_steps=1)
    path = tmp_path / 'tiny.pth'
    digest = gpu.save_state(path, payload)
    restored = gpu.safe_load(path, digest)
    target = torch.nn.Linear(2, 2)
    target.load_state_dict(restored['model'], strict=True)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-4)
    target_optimizer.load_state_dict(restored['optimizer'])
    torch.set_rng_state(restored['cpu_rng'])
    assert fingerprint(target.state_dict()) == fingerprint(payload['model'])
    assert fingerprint(target_optimizer.state_dict()) == fingerprint(payload['optimizer'])
    assert torch.equal(torch.get_rng_state(), payload['cpu_rng'])
    with pytest.raises(FileExistsError):
        gpu.save_state(path, payload)
    with pytest.raises(ValueError, match='SHA-256'):
        gpu.safe_load(path, '0' * 64)


def resume_state():
    return dict(kind=gpu.KIND, config=copy.deepcopy(gpu.CONFIG), completed_steps=1,
                inputs={'revision': 'fixed'}, initializer={'sha256': 'verified'},
                environment={'torch': 'example'}, git_sha='commit')


def test_resume_compatibility():
    state = resume_state()
    def check(candidate, allow=False):
        return gpu.compatible(candidate, state['inputs'], state['initializer'], state['environment'], 'commit', allow)
    assert check(state) == {}
    for key, value in [('config', {}), ('completed_steps', 0), ('completed_steps', 21),
                       ('inputs', {}), ('initializer', {}), ('kind', 'CPU_CHECKPOINT')]:
        bad = copy.deepcopy(state); bad[key] = value
        with pytest.raises(ValueError):
            check(bad)
    changed = copy.deepcopy(state); changed['environment']['torch'] = 'different'
    with pytest.raises(ValueError, match='environment/commit'):
        check(changed)
    assert check(changed, allow=True)['torch'] == ['different', 'example']


@pytest.mark.parametrize('case', ['revision', 'hash', 'split', 'oam', 'duplicate'])
def test_transferred_manifest_rejected_before_pixels(tmp_path, case):
    manifest = json.loads((gpu.REPO / 'docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    if case == 'revision': manifest['revision'] = 'wrong'
    if case == 'hash': manifest['samples'][0]['output_hashes']['rgb.tif'] = '0'*64
    if case == 'split': manifest['samples'][0]['split'] = 'val'
    if case == 'oam': manifest['samples'][0]['oam_id'] = manifest['samples'][1]['oam_id']
    if case == 'duplicate': manifest['samples'][1] = copy.deepcopy(manifest['samples'][0])
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        gpu.validate_inputs(tmp_path, path)


def test_cuda_measurement_api_and_synchronization():
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: calls.append('sync'), memory_allocated=lambda: 1,
        max_memory_allocated=lambda: 2, memory_reserved=lambda: 3, max_memory_reserved=lambda: 4)
    fake = SimpleNamespace(cuda=cuda)
    result, seconds = gpu.timed_cuda(fake, lambda: calls.append('work') or 42)
    assert result == 42 and seconds >= 0 and calls == ['sync', 'work', 'sync']
    assert gpu.cuda_memory(fake) == dict(allocated_bytes=1, max_allocated_bytes=2,
                                        reserved_bytes=3, max_reserved_bytes=4)


def test_cpu_emulated_orchestration_step_resume_restore(tmp_path, monkeypatch):
    """Exercise CLI control flow with tiny CPU convolutions, NOT CUDA verification."""
    import satlaspretrain_models as satlas
    import m1_1_satlas_smoke as shared

    class Tiny(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 2, 1)
            self.fpn = torch.nn.Conv2d(2, 2, 1)
            self.upsample = torch.nn.Conv2d(2, 2, 1)
            self.head = torch.nn.Conv2d(2, 2, 1)
        def to(self, **kwargs):
            return self  # CPU surrogate only
        def forward(self, images, target):
            logits = self.head(self.upsample(self.fpn(self.backbone(images))))
            return logits.softmax(1), torch.nn.functional.cross_entropy(logits, target)

    monkeypatch.setattr(satlas, 'Model', Tiny)
    monkeypatch.setattr(shared, 'checkpoint_digest', lambda *args: None)
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == 'cuda': args = ('cpu', *args[1:])
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(gpu.subprocess, 'check_output',
        lambda command, **kw: '' if command[1] == 'status' else 'emulated-commit\n')
    for name, value in dict(is_available=lambda: True, get_device_properties=lambda index:
        SimpleNamespace(name='CPU_TEST_SURROGATE', total_memory=0), device_count=lambda: 1,
        get_device_capability=lambda index: (0, 0), set_device=lambda index: None,
        manual_seed_all=lambda seed: None, reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, memory_allocated=lambda: 0, max_memory_allocated=lambda: 0,
        memory_reserved=lambda: 0, max_memory_reserved=lambda: 0,
        get_rng_state_all=lambda: [torch.tensor([1, 2], dtype=torch.uint8)],
        set_rng_state_all=lambda states: None).items():
        monkeypatch.setattr(torch.cuda, name, value)

    patches = {}
    for split, size in [('train', 256), ('val', 224)]:
        patches[split] = (torch.ones(1, 3, size, size), torch.zeros(1, size, size, dtype=torch.long),
                         {'valid_fraction.tif': torch.ones(1, size, size).numpy()})
    monkeypatch.setattr(gpu, 'validate_inputs', lambda *args: (patches, {'test': 'synthetic CPU only'}))
    torch.save({'tiny': torch.ones(1)}, tmp_path/'official.pth')
    assert gpu.main(arguments(tmp_path)) == 0
    first = json.loads((tmp_path/'result/result.json').read_text())
    assert first['optimizer_step_status'] == 'GPU_OPTIMIZER_STEP_PASS'
    assert first['completed_steps'] == 1 and all(first['steps'][0]['updated'].values())
    common = ['--data', str(tmp_path/'data'), '--checkpoint', str(tmp_path/'official.pth'),
              '--resume', str(tmp_path/'result/gpu_smoke.pth'),
              '--resume-sha256', first['saved_checkpoint']['sha256']]
    assert gpu.main(common + ['--output', str(tmp_path/'resumed'), '--max-steps', '2']) == 0
    resumed = json.loads((tmp_path/'resumed/result.json').read_text())
    assert resumed['completed_steps'] == 2 and [s['step'] for s in resumed['steps']] == [2]
    assert resumed['initial_eval'] == first['initial_eval']
    assert resumed['restore_status'] == 'GPU_CHECKPOINT_STATE_RESTORED'
    assert gpu.main(common + ['--output', str(tmp_path/'restored'), '--restore-only']) == 0
    restored = json.loads((tmp_path/'restored/result.json').read_text())
    assert restored['status'] == 'GPU_CHECKPOINT_RESTORE_PASS' and restored['completed_steps'] == 1
    assert 'steps' not in restored
