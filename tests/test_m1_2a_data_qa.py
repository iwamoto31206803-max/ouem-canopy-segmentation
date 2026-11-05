import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('dataqa', Path(__file__).parents[1] / 'scripts/m1_2a_data_qa.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def segment(sid, category=1):
    return dict(id=sid, category_id=category, iscrowd=0)


def test_rgb_ids_and_merge():
    rgb = np.array([[[1, 2, 3], [2, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    mask, report = qa.labels(rgb, rgb, [segment(197121), segment(2, 2)])
    assert mask.tolist() == [[1, 1, 0]]
    assert report['zero_pixels'] == 1
    assert report['categories'] == [1, 2]


@pytest.mark.parametrize('segments', [[], [segment(1, 3)]])
def test_unknown_rejected(segments):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    ann = image.copy()
    ann[0, 0, 0] = 1
    with pytest.raises(ValueError):
        qa.labels(image, ann, segments)


def test_dimensions():
    with pytest.raises(ValueError):
        qa.labels(np.zeros((2, 2, 3)), np.zeros((3, 2, 3), dtype=np.uint8), [])


@pytest.mark.parametrize('value', [0, 1])
def test_uniform_preserved(value):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = value
    mask, _ = qa.labels(rgb, rgb, [segment(1)] if value else [])
    assert np.all(mask == value)


def test_selection_and_overlap():
    rows = [dict(image_id=i, oam_id=str(i), validation_fold=f, license='CC-BY 4.0') for i, f in enumerate([2, 8, 0])]
    assert qa.choose(rows, 2) == qa.choose(rows, 2)
    assert [r['split'] for r in qa.choose(rows, 2)] == ['train', 'val']
    rows[2]['oam_id'] = '0'
    with pytest.raises(ValueError, match='overlap'):
        qa.choose(rows, 2)


def test_saved_hashes(tmp_path):
    import io
    from PIL import Image
    raw = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(raw, format='PNG')
    row = dict(height=2, width=2, segments='[]', coco_annotations='[]')
    folder = tmp_path / 'pair'
    hashes, _ = qa.save_pair(folder, row, raw.getvalue(), raw.getvalue())
    for name, expected in hashes.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('Network forbidden in unit tests')
    monkeypatch.setattr(qa.requests, 'get', deny)


def test_range_budget_prevents_request():
    remote = qa.RangeFile('https://example.invalid', 100, qa.Budget(10))
    with pytest.raises(ValueError, match='budget'):
        remote.read(11)
