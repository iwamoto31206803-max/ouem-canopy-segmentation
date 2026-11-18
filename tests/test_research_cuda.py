from collections import Counter
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import research_cuda as cuda
from m1_6_colab_gpu_smoke import safe_load


def synthetic_pool():
    train = [*range(10000, 10069), 614]
    samples = {i: dict(split='train' if i in train else 'val', oam_id=str(i))
               for i in train + list(cuda.VAL)}
    positions = {i: [(9, 1), (1, 9)] if i in train and i != 614 else [(5, 6)]
                 for i in samples}
    return SimpleNamespace(samples=samples, crop_positions=positions)


def test_schedule():
    data = synthetic_pool()
    schedule = cuda.make_schedule(data)
    assert cuda.digest(schedule) == cuda.digest(cuda.make_schedule(data))
    rows = schedule['train']
    assert [r['step'] for r in rows] == list(range(1, 281))
    assert [r['image_id'] for r in schedule['val']] == list(cuda.VAL)
    assert all(r['crop_xywh'] == [5, 6, 224, 224] for r in schedule['val'])
    train = {i for i in data.samples if i not in cuda.VAL}
    for cycle in range(4):
        segment = rows[cycle*70:(cycle+1)*70]
        assert {r['image_id'] for r in segment} == train
        assert all(tuple(r['crop_xywh'][:2]) == sorted(data.crop_positions[r['image_id']])[
            cycle % len(data.crop_positions[r['image_id']])] for r in segment)
    counts = Counter((r['image_id'], tuple(r['crop_xywh'][:2])) for r in rows)
    assert len({(r['image_id'], tuple(r['crop_xywh'])) for r in rows[:140]}) == 139
    assert counts.pop((614, (5, 6))) == 4
    assert set(counts.values()) == {2}
    data.crop_positions[cuda.VAL[0]].append((9, 9))
    with pytest.raises(ValueError, match='Val18'):
        cuda.make_schedule(data)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.layers = nn.Conv2d(3, 2, 1)
        nn.init.zeros_(self.head.layers.weight)
        nn.init.zeros_(self.head.layers.bias)

    def forward(self, images, target):
        logits = self.head.layers(images)
        return logits.softmax(1), nn.functional.cross_entropy(logits, target)


@pytest.mark.parametrize('fail', [False, True])
def test_eval_restores_modes_gradients_and_next_update(fail):
    model = ToyModel().train()
    model.head.eval()
    model.head.layers.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters())
    flags = [m.training for m in model.modules()]
    grads = [p.requires_grad for p in model.parameters()]
    entry = cuda.make_schedule(synthetic_pool())['val'][:1]
    x, y = torch.ones(1, 3, 224, 224), torch.zeros(1, 224, 224, dtype=torch.long)

    def patch(*_args):
        assert torch.is_inference_mode_enabled()
        assert not any(m.training for m in model.modules())
        if fail:
            raise RuntimeError('injected')
        return x, y, {}

    data = SimpleNamespace(patch=patch)
    with pytest.raises(ValueError, match='identity'):
        cuda.evaluate(model, data, [], entry, 'cpu')
    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            cuda.evaluate(model, data, entry, entry, 'cpu')
    else:
        result = cuda.evaluate(model, data, entry, entry, 'cpu')
        assert result['micro']['tree_iou'] is None
        assert result['macro']['valid_image_counts']['precision'] == 0
        assert result['background_only'][0]['fp'] == 0
    assert flags == [m.training for m in model.modules()]
    assert grads == [p.requires_grad for p in model.parameters()]
    assert not model.head.layers._forward_hooks
    assert not optimizer.state
    before = model.head.layers.weight.clone()
    cuda.update(model, optimizer, x, y)
    assert torch.isfinite(model.head.layers.weight.grad).all()
    assert not torch.equal(before, model.head.layers.weight)


def test_checkpoint_copy_restore_and_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _state: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    schedule = cuda.make_schedule(synthetic_pool())
    binding = dict(schedule_sha256=cuda.digest(schedule))
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    x = torch.ones(1, 3, 224, 224)
    y = torch.zeros(1, 224, 224, dtype=torch.long)
    cuda.update(model, optimizer, x, y)
    backup = tmp_path/'backup'
    origin = tmp_path/'checkpoint.pth'
    checkpoint = backup/origin.name
    state = dict(kind=cuda.KIND, binding=binding, schedule=schedule, completed_steps=70,
                 next_item=schedule['train'][70], backup_checkpoint=str(checkpoint),
                 model=model.state_dict(), optimizer=optimizer.state_dict(), deadline=100,
                 cpu_rng=torch.get_rng_state(), python_rng=random.getstate(), cuda_rng=[],
                 fingerprints=cuda.fingerprints(model, optimizer, torch))
    sha = cuda._checkpoint(origin, state)
    assert cuda.copy_backup(origin, sha, backup)['copy_sha_match']
    with pytest.raises(ValueError):
        safe_load(checkpoint, '0'*64)
    loaded = safe_load(checkpoint, sha)
    assert cuda.validate_checkpoint(loaded, binding, schedule) == 70
    assert loaded['next_item']['step'] == 71
    for change in [dict(kind='legacy'), dict(binding={}), dict(next_item=schedule['train'][71]),
                   dict(schedule=dict(schedule, seed=1))]:
        with pytest.raises(ValueError):
            cuda.validate_checkpoint(dict(loaded, **change), binding, schedule)
    restored = ToyModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    assert set(cuda.restore(restored, restored_optimizer, loaded, torch).values()) == {'PASS'}
    assert cuda.fingerprint(restored.state_dict()) == cuda.fingerprint(model.state_dict())
    assert cuda.fingerprint(restored_optimizer.state_dict()) == cuda.fingerprint(optimizer.state_dict())
    with pytest.raises(ValueError, match='original verified backup'):
        cuda.continuation_claim(loaded, origin, sha)
    cuda.continuation_claim(loaded, checkpoint, sha)
    with pytest.raises(FileExistsError):
        cuda.continuation_claim(loaded, checkpoint, sha)


def test_stop_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA runtime required'):
        cuda.run(SimpleNamespace())
    source = tmp_path/'data'
    source.mkdir()
    with pytest.raises(ValueError, match='overlap'):
        cuda.reserve(source/'out', tmp_path/'backup', [source])
    with pytest.raises(FileExistsError):
        cuda.reserve(source, tmp_path/'backup', [])
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match='loss'):
        cuda.update(model, optimizer, torch.full((1, 3, 224, 224), float('nan')),
                    torch.zeros(1, 224, 224, dtype=torch.long))
    assert not optimizer.state
