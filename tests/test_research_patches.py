"""Synthetic direct-window tests; real pilot is a separately authorized CLI run."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import research_patches as patches


def test_positive_area_support_and_direct_windows():
    mask=np.zeros((7,11),bool);mask[2,3]=True
    assert np.array_equal(patches.blocked_grid(mask,(3,5)),patches.reduce_area(mask,(3,5))>0)
    rgb=np.full((8,12,3),120,np.uint8);rgb[:,5]=0
    teacher=np.zeros((8,12),np.uint8)
    chosen=patches.choose_window(rgb,teacher,(8,12),1054,0,3)
    assert chosen==patches.choose_window(rgb,teacher,(8,12),1054,0,3)
    assert chosen['placement_count']==42  # Both sides of the black column participate.
    assert chosen['selected_black_support_cells']==0
    assert len({tuple(patches.choose_window(rgb,teacher,(8,12),1054,s,3)['crop_xywh']) for s in range(5)})>1


def test_internal_dark_and_white_are_not_automatic_nodata():
    rgb=np.full((8,12,3),255,np.uint8);rgb[3,3]=0
    result=patches.choose_window(rgb,np.ones((8,12),np.uint8),(8,12),1054,size=3)
    assert result['placement_count']==60 and result['bw_placement_count']==0
    assert result['black']['border_connected_pixels']==0
    with pytest.raises(ValueError):patches.choose_window(np.zeros_like(rgb),np.zeros((8,12),np.uint8),(8,12),1054,size=3)
    with pytest.raises(ValueError):patches.choose_window(rgb,np.zeros((8,12),np.uint8),(2,2),1054,size=3)


@pytest.mark.parametrize('value',[0.,1.,.49999999,.5,.50000001])
def test_saved_fraction_threshold_and_shifted_grid(tmp_path,value):
    rgb=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3)
    teacher=np.full((6,7),value,np.float64)
    grid=from_origin(100,200,.5,.5)
    arrays,transform=patches.crop_arrays(rgb,teacher,(6,7),grid,[2,1,3,3])
    assert transform==grid*rasterio.Affine.translation(2,1)
    assert np.array_equal(arrays['rgb.tif'],rgb[1:4,2:5].transpose(2,0,1))
    assert np.all(arrays['canopy_mask.tif']==int(np.float32(value)>=.5))
    patches.write_products(tmp_path/'crop',arrays,'EPSG:3395',transform)


@pytest.fixture
def inputs(tmp_path):
    root=tmp_path/'source';evidence=tmp_path/'evidence';evidence.mkdir()
    (root/'archives').mkdir(parents=True)
    rows=[];prior=[]
    for i,(rid,split) in enumerate(patches.TARGETS.items()):
        oam=f'synthetic-{rid}';stem=f'{oam}_{rid}'
        row=dict(image_id=rid,oam_id=oam,height=20,width=20,license='CC-BY 4.0',
            validation_fold=0 if split=='val' else 1,crs='EPSG:3395',bounds=[0,0,2,2],
            meta=f'https://invalid.example/{oam}_meta.json')
        rgb=root/f'extracted/dataset/holdout/images/{stem}.tif';rgb.parent.mkdir(parents=True,exist_ok=True)
        with rasterio.open(rgb,'w',driver='GTiff',height=20,width=20,count=3,dtype='uint8',
            crs='EPSG:3395',transform=from_origin(0,2,.1,.1)) as dst:dst.write(np.full((3,20,20),120,np.uint8))
        ann=root/f'extracted/dataset/holdout/masks/{stem}.png';ann.parent.mkdir(parents=True,exist_ok=True)
        label=np.zeros((20,20,3),np.uint8)
        if i%2:label[:,:,0]=2
        Image.fromarray(label).save(ann)
        stac=dict(id=oam,properties={'license':'CC-BY-4.0','oam:producer_name':'SYNTHETIC',
            'providers':[dict(name='SYNTHETIC',roles=['licensor'])]},assets={'metadata':dict(href=row['meta'])})
        (evidence/(oam+'-stac.json')).write_text(json.dumps(stac),encoding='utf-8')
        rows.append(row)
        prior.append(dict(image_id=rid,split=split,input_sha256=dict(rgb=patches.sha(rgb),annotation=patches.sha(ann)),
            geometry=dict(output_shape=[4,4]),scenarios={'edge_black_only':{'patches':{'3':{'placement_count':4}}}}))
    (root/'archives/train_meta.json').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    (root/'archives/test_meta.json').write_text('',encoding='utf-8')
    diagnosis=tmp_path/'phaseb.json';diagnosis.write_text(json.dumps(dict(images=prior)),encoding='utf-8')
    return root,diagnosis,evidence


def run(inputs,output):
    root,diagnosis,evidence=inputs
    return patches.prepare(root,diagnosis,patches.sha(diagnosis),evidence,output,size=3)


def test_five_pair_roundtrip_repeat_and_sources_preserved(inputs,tmp_path):
    root,diagnosis,evidence=inputs
    before={str(p):patches.sha(p) for p in root.rglob('*') if p.is_file()}
    first=run(inputs,tmp_path/'first');second=run(inputs,tmp_path/'repeat')
    assert first['status']=='EXPLORATORY_PILOT_PRODUCTS_COMPLETE' and len(first['samples'])==5
    for i,(a,b) in enumerate(zip(first['samples'],second['samples'])):
        assert a['crop_xywh']==b['crop_xywh'] and a['products']==b['products']
        assert a['patch_grid']['shape']==[3,3] and a['selected_black_support_cells']==0
        assert a['canopy_mask_fraction']==i%2
        path=tmp_path/'first'/a['products']['canopy_mask.tif']['path']
        with rasterio.open(path) as src:assert src.transform==rasterio.Affine(*a['patch_grid']['affine'])
    assert {str(p):patches.sha(p) for p in root.rglob('*') if p.is_file()}==before
    with pytest.raises(ValueError):run(inputs,tmp_path/'first')
    with pytest.raises(ValueError):run(inputs,root/'output')


@pytest.mark.parametrize('case',['hash','license','split','format','count'])
def test_invalid_inputs_stop(inputs,tmp_path,case):
    root,diagnosis,evidence=inputs
    if case=='hash':
        with pytest.raises(ValueError):patches.prepare(root,diagnosis,'0'*64,evidence,tmp_path/'out',size=3)
        return
    if case in ('license','split'):
        path=root/'archives/train_meta.json';rows=[json.loads(l) for l in path.read_text().splitlines()]
        rows[0]['license' if case=='license' else 'validation_fold']='unknown' if case=='license' else 0
        path.write_text('\n'.join(json.dumps(r) for r in rows))
    elif case=='format':
        path=next((root/'extracted/dataset/holdout/images').glob('*.tif'))
        with rasterio.open(path) as src:profile=src.profile
        profile['count']=1
        with rasterio.open(path,'w',**profile) as dst:dst.write(np.ones((1,20,20),np.uint8))
    else:
        value=json.loads(diagnosis.read_text());value['images'][0]['scenarios']['edge_black_only']['patches']['3']['placement_count']=99
        diagnosis.write_text(json.dumps(value))
    with pytest.raises(ValueError):run(inputs,tmp_path/'out')
    if (tmp_path/'out/crop_plan.json').exists():
        assert json.loads((tmp_path/'out/crop_plan.json').read_text())['status']=='FAIL'
