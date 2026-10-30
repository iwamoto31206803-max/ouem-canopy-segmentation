import numpy as np
import pytest
from PIL import Image
from canopy_prep.core import binary_canopy_mask, degrade_pair, validate_disjoint, write_pair

def test_crop_fraction():
    rgb = np.full((12, 12, 3), 127, dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:3, :5] = 1
    low, frac, binary = degrade_pair(rgb, mask)
    assert low.shape == (2, 2, 3)
    assert frac.shape == binary.shape == (2, 2)
    assert np.isclose(frac[0,0], 0.6)
    assert binary.sum() == 1

def test_uniform_masks():
    rgb = np.zeros((10,10,3), dtype=np.uint8)
    for value in (0,1):
        _, frac, binary = degrade_pair(rgb, np.full((10,10), value, dtype=np.uint8))
        assert np.all(frac == value) and np.all(binary == value)

def test_explicit_mapping():
    assert binary_canopy_mask(np.array([[0,1,2,3]]), {1,2}).tolist() == [[0,1,1,0]]
    with pytest.raises(ValueError):
        binary_canopy_mask(np.array([[0]]), set())

def test_split_leakage():
    with pytest.raises(ValueError):
        validate_disjoint([{"oam_id":"x","split":"train"},{"oam_id":"x","split":"validation"}])

def test_write(tmp_path):
    rgb = np.zeros((10,10,3), dtype=np.uint8)
    mask = np.ones((10,10), dtype=np.uint8)
    record = write_pair(rgb, mask, tmp_path, source_id="scene-01", oam_id="oam-1", split="train")
    assert len(record["files"]["rgb"]["sha256"]) == 64
    assert np.array(Image.open(tmp_path / record["files"]["mask"]["path"])).min() == 255
    with pytest.raises(FileExistsError):
        write_pair(rgb, mask, tmp_path, source_id="scene-01", oam_id="oam-1", split="train")
