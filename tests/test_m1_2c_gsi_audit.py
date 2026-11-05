from pathlib import Path
import sys
import math
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from m1_2c_gsi_audit import grid, tile, ground


def test_xyz_origin_and_spacing():
    assert tile(0, 0, 1) == (1, 1)
    r, x, y = grid(0, 0, 18)
    assert r == pytest.approx(0.5971642834779395)
    _, nx, ny = grid(1, 1, 18)
    assert nx-x == pytest.approx(256*r)
    assert y-ny == pytest.approx(256*r)


def test_ground_distance_is_smaller_than_map_spacing():
    r, _, _ = grid(0, 0)
    g = ground(35.63466,139.5782,r)
    assert 0.48 < g['east_m'] < 0.49
    assert 0.48 < g['north_m'] < 0.49
    assert abs(g['east_m'] - g['spherical_approx_m']) < 0.002
    assert abs(g['north_m'] - g['spherical_approx_m']) < 0.002
