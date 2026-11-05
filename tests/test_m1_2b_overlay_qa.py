import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from m1_2b_overlay_qa import overlay


def test_overlay_preserves_grid_background_and_input():
    rgb = np.full((2, 3, 3), 100, dtype=np.uint8)
    original = rgb.copy()
    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 1] = 1
    result = overlay(rgb, mask)
    assert result.shape == rgb.shape
    assert np.array_equal(rgb, original)
    assert np.array_equal(result[mask == 0], rgb[mask == 0])
    assert not np.array_equal(result[0, 1], rgb[0, 1])
    with pytest.raises(ValueError):
        overlay(rgb, np.zeros((3, 2)))


def test_rle_fortran_order_and_length():
    from m1_2b_overlay_qa import decode_rle
    assert decode_rle({'size': [2, 2], 'counts': [1, 2, 1]}).tolist() == [[0, 1], [1, 0]]
    with pytest.raises(ValueError):
        decode_rle({'size': [2, 2], 'counts': [1, 2]})
