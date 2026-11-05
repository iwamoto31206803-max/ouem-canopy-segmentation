from pathlib import Path
import sys
import socket
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_3a_loader import load_patch, check_split
from m1_2d_geometry_audit import sha


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Network forbidden'))


@pytest.fixture
def pair(tmp_path):
    folder=tmp_path/'train'/'661'; folder.mkdir(parents=True)
    transform=from_origin(100,200,0.5,0.5)
    mask=np.zeros((1,8,8),dtype=np.uint8); mask[:,2:4,1:3]=1
    rgb=np.repeat(mask*255,3,axis=0)
    arrays={'rgb.tif':rgb,'canopy_mask.tif':mask,'canopy_fraction.tif':mask.astype(np.float32),
            'valid_fraction.tif':np.ones((1,8,8),dtype=np.float32)}
    sample=dict(image_id=661,split='train',output_shape=[8,8],output_crs='EPSG:3395',
                output_transform=list(transform)[:6],output_bounds=[100,196,104,200],output_hashes={})
    for name,array in arrays.items():
        with rasterio.open(folder/name,'w',driver='GTiff',width=8,height=8,count=array.shape[0],dtype=array.dtype,
                           crs=sample['output_crs'],transform=transform) as dst: dst.write(array)
        sample['output_hashes'][name]=sha(folder/name)
    return tmp_path,sample,folder


def test_deterministic_patch_and_alignment(pair):
    root,sample,_=pair
    x,y,crop=load_patch(root,sample,'train',4)
    x2,y2,_=load_patch(root,sample,'train',4)
    import torch
    assert torch.equal(x,x2) and torch.equal(y,y2)
    assert torch.equal(x[:,0],y.float())
    assert x.dtype==torch.float32 and y.dtype==torch.long
    assert x.shape==(1,3,4,4) and y.shape==(1,4,4)
    assert all(a.shape[1:]==(4,4) for a in crop.values())


def test_hash_failure(pair):
    root,sample,folder=pair
    (folder/'rgb.tif').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Hash'): load_patch(root,sample,'train',4)


@pytest.mark.parametrize('case',['grid','label','dtype','fraction','valid','shape'])
def test_invalid_data_rejected(pair,case):
    root,sample,folder=pair
    name='canopy_mask.tif' if case in ['label','dtype','shape'] else 'canopy_fraction.tif' if case=='fraction' else 'valid_fraction.tif' if case=='valid' else 'rgb.tif'
    path=folder/name
    with rasterio.open(path) as src:
        data=src.read(); profile=src.profile
    if case=='grid': profile['transform']=from_origin(101,200,.5,.5)
    if case=='label': data.fill(2)
    if case=='dtype': data=data.astype(np.int16); profile['dtype']='int16'
    if case=='fraction': data.fill(np.nan)
    if case=='valid': data[:,:,0]=0
    if case=='shape': data=np.repeat(data,2,axis=0); profile['count']=2
    with rasterio.open(path,'w',**profile) as dst: dst.write(data)
    sample['output_hashes'][name]=sha(path)
    with pytest.raises(ValueError): load_patch(root,sample,'train',4)


def test_no_padding_or_split_mixing(pair):
    root,sample,_=pair
    with pytest.raises(ValueError,match='Padding'): load_patch(root,sample,'train',16)
    with pytest.raises(ValueError,match='split'): load_patch(root,sample,'val',4)
    for size in [0,-1,1.5]:
        with pytest.raises(ValueError): load_patch(root,sample,'train',size)


def test_official_ignore_default_is_zero_but_mean_diluted():
    import torch
    from satlaspretrain_models.models.heads import SimpleHead
    head=SimpleHead('segment',[[1,2]],2)
    features=torch.ones(1,2,2,2)
    target=torch.tensor([[[0,-100],[1,-100]]])
    logits=head.layers(features)
    _,loss=head(torch.zeros(1,3,2,2),[features],target)
    raw=torch.nn.functional.cross_entropy(logits,target,reduction='none')
    assert raw[0,0,1]==0 and raw[0,1,1]==0
    assert torch.allclose(loss,raw.sum()/4)
