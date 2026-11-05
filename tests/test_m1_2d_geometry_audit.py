from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_2d_geometry_audit import intervals, require_hash, sha


def test_world_mercator_ground_scale():
    from pyproj import Transformer
    x,y=Transformer.from_crs(4326,3395,always_xy=True).transform(31.418,50.032)
    result=intervals(3395,x,y,0.1,0.1)
    assert result['east_m']==pytest.approx(0.06436,abs=0.00001)
    assert result['north_m']==pytest.approx(result['east_m'],abs=1e-7)
    assert result['east_m']*5 < 0.484


def test_hash_rejects_changed_original(tmp_path):
    p=tmp_path/'fixture'; p.write_bytes(b'original')
    expected=sha(p)
    assert require_hash(p,expected)==expected
    p.write_bytes(b'changed')
    with pytest.raises(ValueError): require_hash(p,expected)
