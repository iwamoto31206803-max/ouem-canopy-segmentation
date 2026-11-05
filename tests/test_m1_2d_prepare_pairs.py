from pathlib import Path
import sys
import numpy as np
import pytest
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_2d_prepare_pairs import area_weights, reduce_area, binary, output_grid
from m1_2d_geometry_audit import audit


def test_exact_area_noninteger_reduction():
    assert np.allclose(area_weights(3,2),[[2/3,1/3,0],[0,1/3,2/3]])
    a=np.array([[0,1,1],[0,1,1]],dtype=np.uint8)
    result=reduce_area(a,(1,2))
    assert np.allclose(result,[[1/3,1]])
    assert result.mean()==pytest.approx(a.mean())


@pytest.mark.parametrize('value',[0,1])
def test_uniform_fraction_and_validity(value):
    result=reduce_area(np.full((13,11),value,dtype=np.uint8),(3,2))
    assert np.allclose(result,value)
    assert np.all(binary(result)==value)


def test_threshold_and_rgb_determinism():
    assert binary(np.array([0.499999,0.5,0.500001])).tolist()==[0,1,1]
    rgb=np.arange(12*10*3,dtype=np.uint16).reshape(12,10,3)%256
    assert np.array_equal(reduce_area(rgb,(3,4)),reduce_area(rgb,(3,4)))
    assert reduce_area(rgb,(3,4)).shape==(3,4,3)


def test_grid_extent_and_ground_spacing():
    from pyproj import Transformer
    x,y=Transformer.from_crs(4326,3395,always_xy=True).transform(31.418,50.032)
    src=from_origin(x,y,0.1,0.1)
    shape,grid,gsd=output_grid(3395,src,2048,2048)
    assert shape==(264,264)
    assert grid*(0,0)==src*(0,0)
    assert np.allclose(grid*(shape[1],shape[0]),src*(2048,2048),rtol=0,atol=1e-8)
    assert abs(gsd['east_m']-.5)<.002
    assert abs(gsd['north_m']-.5)<.002


def test_wrong_assigned_split_rejected_before_inputs(tmp_path):
    # Split identity is checked before a sample's source files are read.
    manifest={'samples':[{'image_id':661,'oam_id':'same','split':'val'},
                         {'image_id':1972,'oam_id':'same','split':'val'}]}
    with pytest.raises(ValueError,match='Split/source-group'):
        audit(tmp_path,manifest)


def test_reject_upscale():
    with pytest.raises(ValueError): area_weights(2,3)
