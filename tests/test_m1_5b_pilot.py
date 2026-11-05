"""Offline pilot contracts, synthetic GeoTIFF/panoptic pairs and mocked HTTP."""
import copy
import io
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import m1_5b_pilot as pilot


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:pytest.fail('Network forbidden'))


@pytest.fixture
def source(tmp_path):
    folder=tmp_path/'cache'/'raw'/'2'; folder.mkdir(parents=True)
    transform=from_origin(100,200,.1,.1)
    rgb=np.full((3,10,10),100,dtype=np.uint8)
    with rasterio.open(folder/'rgb.bin','w',driver='GTiff',height=10,width=10,count=3,dtype='uint8',
                       crs=3395,transform=transform) as dst: dst.write(rgb)
    annotation=np.zeros((10,10,3),dtype=np.uint8); annotation[:5,:5,0]=1; annotation[5:,5:,1]=1
    Image.fromarray(annotation).save(folder/'annotation.bin',format='PNG')
    segments=[dict(id=1,category_id=1,iscrowd=0),dict(id=256,category_id=2,iscrowd=0)]
    (folder/'segments.json').write_text(json.dumps(segments)); (folder/'coco_annotations.json').write_text('[]')
    mask,qa=pilot.labels(rgb.transpose(1,2,0),annotation,segments)
    Image.fromarray(mask).save(folder/'candidate-mask.png')
    Image.fromarray(rgb.transpose(1,2,0)).save(folder/'preview.jpg')
    stac=dict(id='source',properties={'license':'CC-BY-4.0'},bbox=[-1,-1,1,1])
    (folder/'stac.json').write_text(json.dumps(stac))
    row=dict(image_id=2,oam_id='source',height=10,width=10,bounds=[100,199,101,200],
             license='CC-BY 4.0',official_split='train',validation_fold=1,split='train')
    hashes={p.name:pilot.sha(p) for p in folder.iterdir()}
    return folder,dict(metadata=row,hashes=hashes,label_qa=qa)


def test_labels_geometry_area_and_determinism(source):
    folder,sample=source
    arrays,info=pilot.inspect_source(folder,sample)
    again,info2=pilot.inspect_source(folder,sample)
    assert info==info2 and all(np.array_equal(arrays[k],again[k]) for k in arrays)
    assert set(arrays)==pilot.PRODUCTS and info['output_shape']==[2,2]
    assert info['source_bounds']==info['output_bounds']==[100,199,101,200]
    assert np.array_equal(arrays['canopy_fraction.tif'][0],[[1,0],[0,1]])
    assert np.array_equal(arrays['canopy_mask.tif'],arrays['canopy_fraction.tif'])
    assert np.all(arrays['valid_fraction.tif']==1)
    assert abs(info['output_ground_gsd']['east_m']-.5)<.001
    assert sample['label_qa']['categories']==[1,2]
    assert pilot.binary(np.array([.499,.5,.501])).tolist()==[0,1,1]


@pytest.mark.parametrize('value',[0,1])
def test_uniform_masks_preserved(source,value):
    folder,sample=source
    annotation=np.full((10,10,3),0,dtype=np.uint8); annotation[:,:,0]=value
    Image.fromarray(annotation).save(folder/'annotation.bin',format='PNG')
    segments=[] if value==0 else [dict(id=1,category_id=1,iscrowd=1)]
    (folder/'segments.json').write_text(json.dumps(segments))
    Image.fromarray(np.full((10,10),value,dtype=np.uint8)).save(folder/'candidate-mask.png')
    sample['hashes']={p.name:pilot.sha(p) for p in folder.iterdir()}
    products,info=pilot.inspect_source(folder,sample)
    assert np.all(products['canopy_fraction.tif']==value) and np.all(products['canopy_mask.tif']==value)
    assert info['uniform_background']==(value==0)


@pytest.mark.parametrize('case',['hash','unknown_id','unknown_category','dimensions','affine','nodata','bounds'])
def test_bad_sources_rejected(source,case):
    folder,sample=source
    if case=='hash': (folder/'segments.json').write_text('tampered')
    if case=='unknown_id': Image.fromarray(np.full((10,10,3),3,dtype=np.uint8)).save(folder/'annotation.bin',format='PNG')
    if case=='unknown_category': (folder/'segments.json').write_text('[{"id":1,"category_id":3,"iscrowd":0}]')
    if case=='dimensions': sample['metadata']['width']=11
    if case=='bounds': sample['metadata']['bounds']=[101,199,102,200]
    if case in ['affine','nodata']:
        with rasterio.open(folder/'rgb.bin','r+') as dst:
            if case=='affine': dst.transform=from_origin(101,200,.1,.1)
            else: dst.nodata=0
    if case!='hash': sample['hashes']={p.name:pilot.sha(p) for p in folder.iterdir()}
    with pytest.raises(ValueError): pilot.inspect_source(folder,sample)


def test_unmarked_black_border_is_hold_not_background(source):
    folder,sample=source
    with rasterio.open(folder/'rgb.bin','r+') as dst:
        array=dst.read(); array[:,:,0]=0; dst.write(array)
    sample['hashes']['rgb.bin']=pilot.sha(folder/'rgb.bin')
    arrays,info=pilot.inspect_source(folder,sample)
    assert arrays=={} and info['source_validity_hold']=='HOLD_UNMARKED_BLACK_BORDER_VALIDITY'
    assert info['black_border_pixels']>0 and info['gdal_all_valid']


def test_output_readback_and_hash_grid_rejection(source,tmp_path):
    folder,sample=source
    arrays,info=pilot.inspect_source(folder,sample)
    info.update(image_id=2,split='train',output_hashes={})
    output=tmp_path/'products'; target=output/'train'/'2'; target.mkdir(parents=True)
    from affine import Affine
    for name,array in arrays.items():
        with rasterio.open(target/name,'w',driver='GTiff',height=2,width=2,count=array.shape[0],dtype=array.dtype,
                           crs=info['output_crs'],transform=Affine(*info['output_transform'])) as dst: dst.write(array)
        info['output_hashes'][name]=pilot.sha(target/name)
    pilot.verify_sample(output,info,arrays)
    wrong=copy.deepcopy(info); wrong['output_transform'][2]+=1
    with pytest.raises(ValueError,match='grid'): pilot.verify_sample(output,wrong)
    (target/'rgb.tif').write_bytes(b'broken')
    with pytest.raises(ValueError,match='Hash'): pilot.verify_sample(output,info)


def test_candidate_restriction_and_output_protection(tmp_path):
    for ids in [[1],[661],[999],[2,2],[]]:
        with pytest.raises(ValueError,match='candidates'): pilot.load_inventory(tmp_path,ids)
    protected=tmp_path/'old'; protected.mkdir(); (protected/'keep').write_text('keep')
    for dest in [protected,protected/'new',pilot.REPO/'new-data']:
        with pytest.raises(ValueError): pilot.fresh(dest,[protected])
    assert (protected/'keep').read_text()=='keep'
    assert pilot.main(['plan','--inventory-cache',str(protected),'--cache',str(protected)])==1
    assert not (protected/'plan-execution.json').exists()
    with pytest.raises(SystemExit): pilot.main(['plan','--inventory-cache',str(protected),'--cache',str(tmp_path/'new'),
                                               '--transfer-mib','257'])


def test_split_overlap_proximity_and_holdout():
    def row(i,fold,bounds,group):
        return dict(image_id=i,oam_id=group,official_split='train',license='CC-BY 4.0',validation_fold=fold,
                    wgs84_bounds=bounds,center=[(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2])
    a=row(2,1,[0,0,.001,.001],'a'); b=row(9,0,[1,1,1.001,1.001],'b')
    assert pilot.separation(a,[a,b])['nearest_opposite_footprint_lower_bound_m']>1000
    for bad in [dict(b,oam_id='a'),dict(b,wgs84_bounds=a['wgs84_bounds']),
                dict(b,wgs84_bounds=[.002,0,.003,.001],center=[.0025,.0005])]:
        with pytest.raises(ValueError,match='HOLD'): pilot.separation(a,[a,bad])
    for bad in [dict(a,official_split='test'),dict(a,validation_fold=-1),dict(a,license='unknown')]:
        with pytest.raises(ValueError,match='HOLD'): pilot.separation(bad,[bad,b])


def test_rights_missing_and_review_bound_to_artifacts(source,tmp_path):
    folder,sample=source
    row=sample['metadata']; row['source_evidence']={'snapshot_sha256':sample['hashes']['stac.json'],'url':'https://example.invalid'}
    row['meta']='https://example.invalid/original.json'
    raw=folder/'stac.json'; destination=tmp_path/'source-stac.json'; destination.write_bytes(raw.read_bytes())
    with pytest.raises(ValueError,match='ATTRIBUTION'): pilot.rights(row,tmp_path)
    assert not pilot.review_gate(sample,{})
    review=dict(preview_sha256=sample['hashes']['preview.jpg'],stac_sha256=sample['hashes']['stac.json'],
                alignment='PASS',quality='PASS',attribution='PASS')
    assert pilot.review_gate(sample,{'2':review})
    review['preview_sha256']='wrong'
    assert not pilot.review_gate(sample,{'2':review})


def test_range_chunks_row_group_and_no_implicit_fetch():
    table=pa.table({'image_id':[2,3,9],'image':[b'a',b'b',b'c'],'annotation':[b'd',b'e',b'f']})
    stream=io.BytesIO(); pq.write_table(table,stream,row_group_size=2)
    data=stream.getvalue(); metadata=pq.read_metadata(io.BytesIO(data))
    assert pilot.group_for(metadata,2)==(1,0)
    pieces=pilot.column_ranges(metadata,0)
    chunks=[(p['start'],data[p['start']:p['start']+p['size']]) for p in pieces]
    source=pilot.Chunks(len(data),chunks)
    parquet=pq.ParquetFile(source,metadata=metadata,pre_buffer=False)
    assert parquet.read_row_group(0,columns=['image_id','image','annotation']).to_pylist()==table.slice(0,2).to_pylist()
    source.seek(len(data)-1)
    with pytest.raises(ValueError,match='unplanned'): source.read(1)


def test_network_budget_incomplete_range_and_cache(tmp_path):
    class Response:
        status_code=206; url='https://example.invalid'
        def __init__(self,body,length=4):
            self.raw=io.BytesIO(body); self.headers={'Content-Range':'bytes 0-3/4','Content-Length':str(length)}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
    network=pilot.Network(tmp_path,4,2*pilot.MIB)
    network.session=SimpleNamespace(get=lambda *a,**k:Response(b'abcd'))
    assert network.cached('https://example.invalid',0,4)==b'abcd'
    assert network.cached('https://example.invalid',0,4)==b'abcd' and network.log['bytes']==4
    with pytest.raises(ValueError,match='budget'): network.get('https://example.invalid',0,1)
    assert network.log['bytes']==4
    other=tmp_path/'other'; other.mkdir()
    network=pilot.Network(other,4,2*pilot.MIB)
    network.session=SimpleNamespace(get=lambda *a,**k:Response(b'ab'))
    with pytest.raises(ValueError,match='Incomplete'): network.get('https://example.invalid',0,4)
    assert network.log['bytes']==2
    with pytest.raises(ValueError,match='Storage'): pilot.storage_gate(tmp_path,None,3*pilot.MIB,2*pilot.MIB)



def test_budget_selection_determinism_and_reduction():
    rows=[dict(image_id=i,height=2048,width=2048) for i in [2,3,9]]
    groups=[dict(targets=[{'image_id':i}],compressed_bytes=75*pilot.MIB) for i in [2,3,9]]
    expected=pilot.bounded_selection(rows,groups,256*pilot.MIB,127*pilot.MIB)
    assert expected==([2,9],{3:'HOLD_STORAGE_ESTIMATE'})
    assert pilot.bounded_selection(rows[::-1],groups[::-1],256*pilot.MIB,127*pilot.MIB)==expected
    selected,held=pilot.bounded_selection(rows,groups,74*pilot.MIB,127*pilot.MIB)
    assert not selected and set(held.values())=={'HOLD_TRANSFER_ESTIMATE'}


def test_prepare_offline_manifest_and_readback(source,tmp_path,monkeypatch):
    folder,sample=source; cache=folder.parents[1]; output=tmp_path/'products'
    raw=dict(revision=pilot.REVISION,status='RAW_ACQUIRED_QA_PENDING',samples=[sample])
    pilot.dump(cache/'raw_manifest.json',raw)
    pilot.dump(cache/'plan.json',dict(storage_limit_bytes=128*pilot.MIB))
    pilot.dump(cache/'network.json',dict(bytes=0))
    monkeypatch.setattr(pilot,'load_inventory',lambda *a:([sample['metadata']],[],{}))
    monkeypatch.setattr(pilot,'rights',lambda *a:({},dict(stac_sha256=sample['hashes']['stac.json'])))
    monkeypatch.setattr(pilot,'separation',lambda *a:{'synthetic':'no network'})
    args=SimpleNamespace(inventory_cache=tmp_path/'inventory',cache=cache,output=output,ids=[2],reviews=None,storage_mib=128)
    result=pilot.prepare(args,dict(revision=pilot.REVISION))
    assert result['status']=='REAL_PILOT_NOT_VERIFIED'  # no Val and no manual review
    assert result['samples'][0]['status']=='PAIR_GENERATED_REVIEW_PENDING'
    assert not result['samples'][0]['training_eligible']
    assert set(result['samples'][0]['output_hashes'])==pilot.PRODUCTS
    pilot.verify_sample(output,result['samples'][0])
    again=tmp_path/'again'; args.output=again
    second=pilot.prepare(args,dict(revision=pilot.REVISION))
    assert second['samples'][0]['output_hashes']==result['samples'][0]['output_hashes']
    args.output=output
    with pytest.raises(ValueError,match='fresh'): pilot.prepare(args,{})


def test_original_object_cross_split_rejected():
    a=dict(image_id=2,oam_id='one',license='CC-BY 4.0',official_split='train',validation_fold=2,
           meta='https://oin-hotosm.s3.amazonaws.com/source/file_meta.json',
           wgs84_bounds=[0,0,.001,.001],center=[.0005,.0005])
    b=dict(a,image_id=9,oam_id='two',validation_fold=0,center=[20,20],wgs84_bounds=[20,20,20.001,20.001],
           meta='https://oin-hotosm-temp.s3.amazonaws.com/source/file_meta.json')
    with pytest.raises(ValueError,match='ORIGINAL_OBJECT'): pilot.separation(a,[a,b])


def test_unknown_source_license_is_hold(source,tmp_path):
    folder,sample=source
    item=json.loads((folder/'stac.json').read_text()); item['properties']['license']='unknown'
    path=tmp_path/'source-stac.json'; pilot.dump(path,item)
    row=dict(sample['metadata'],source_evidence={'snapshot_sha256':pilot.sha(path)})
    with pytest.raises(ValueError,match='HOLD_SOURCE_LICENSE'): pilot.rights(row,tmp_path)


def test_acquire_rejects_holdout_plan_before_network(tmp_path,monkeypatch):
    plan=dict(revision=pilot.REVISION,ids=[2],status='PLAN_READY',transfer_limit_bytes=256*pilot.MIB,
              storage_limit_bytes=128*pilot.MIB,accepted_ids=[2],groups=[dict(shard='data/test-00000.parquet',
              shard_size=10,acquire_targets=[{'image_id':2,'index':0}])])
    pilot.dump(tmp_path/'plan.json',plan)
    pilot.dump(tmp_path/'publisher.json',dict(sha=pilot.REVISION,siblings=[dict(rfilename='data/test-00000.parquet',size=10)]))
    monkeypatch.setattr(pilot,'load_inventory',lambda *a:([dict(image_id=2,shard='data/train-00000.parquet')],[],{}))
    monkeypatch.setattr(pilot,'Network',lambda *a:pytest.fail('Network must not be created'))
    args=SimpleNamespace(cache=tmp_path,inventory_cache=tmp_path/'metadata',ids=[2],transfer_mib=256,storage_mib=128)
    with pytest.raises(ValueError,match='Holdout'): pilot.acquire(args,{})
    assert not (tmp_path/'raw').exists()


@pytest.mark.parametrize('status',['REAL_PILOT_NOT_VERIFIED','OFFLINE_REVERIFICATION_COMPLETE_HOLDS'])
def test_hold_is_nonzero_exit(tmp_path,monkeypatch,status):
    monkeypatch.setattr(pilot,'prepare',lambda args,report:report.update(status=status))
    assert pilot.main(['prepare','--inventory-cache',str(tmp_path/'meta'),'--cache',str(tmp_path/'cache'),
                       '--output',str(tmp_path/'output')])==3


def test_border_audit_connectivity_and_internal_components():
    from m1_5b2_border_audit import components,component_report
    diagonal=np.eye(3,dtype=bool)
    assert len(np.unique(components(diagonal,4)))-1==3
    assert len(np.unique(components(diagonal,8)))-1==1
    mask=np.zeros((8,8),dtype=bool); mask[:,0]=True; mask[3,4]=True
    stats,border,interior=component_report(mask,np.ones((8,8),dtype=np.uint8))
    assert stats['components']==2 and stats['border_connected_pixels']==8
    assert stats['interior_pixels']==1 and stats['interior_positive_label_pixels']==1
    assert stats['largest'][0]['bbox_xyxy']==[0,0,1,8]
    assert not np.any(border & interior)


def test_border_audit_union_find_against_independent_flood_fill():
    from m1_5b2_border_audit import components
    from collections import deque
    rng=np.random.default_rng(0)
    for connectivity in [4,8]:
        for _ in range(10):
            mask=rng.random((12,13))<.4
            expected=np.zeros(mask.shape,dtype=int); number=0
            for y,x in np.argwhere(mask):
                if expected[y,x]: continue
                number+=1; queue=deque([(y,x)]); expected[y,x]=number
                while queue:
                    r,c=queue.popleft()
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]+([] if connectivity==4 else [(-1,-1),(-1,1),(1,-1),(1,1)]):
                        a,b=r+dy,c+dx
                        if 0<=a<12 and 0<=b<13 and mask[a,b] and not expected[a,b]:
                            expected[a,b]=number; queue.append((a,b))
            observed=components(mask,connectivity)
            for region in range(1,number+1):
                ids=np.unique(observed[expected==region]); assert len(ids)==1 and ids[0]!=0
                assert np.array_equal(observed==ids[0],expected==region)


def test_border_audit_square_is_geometric_not_validity():
    from m1_5b2_border_audit import largest_clear_square,analyze
    blocked=np.zeros((5,7),dtype=bool); blocked[:,:3]=True
    assert largest_clear_square(blocked)['side_px']==4
    assert largest_clear_square(np.ones((3,3),dtype=bool))['side_px']==0
    rgb=np.full((6,6,3),100,dtype=np.uint8); rgb[:,0]=0; rgb[:,1]=3; rgb[3,4]=0
    result,*_=analyze(rgb,np.ones((6,6),dtype=np.uint8))
    assert result['exact_4']['border_connected_pixels']==6
    assert result['exact_4']['interior_positive_label_pixels']==1
    assert result['near_black_sensitivity'][0]['border_connected_pixels']==12
    assert result['hypothetical_clear_square']['side_px']==4
    assert 'not a measured valid' in result['nonblack_meaning']


def test_border_audit_empty_black_and_bad_inputs():
    from m1_5b2_border_audit import analyze,components
    result,*_=analyze(np.full((4,4,3),100,dtype=np.uint8),np.zeros((4,4),dtype=np.uint8))
    assert result['exact_4']['components']==0 and result['boundary']['adjacent_nonblack_pixels']==0
    assert result['hypothetical_clear_square']['side_px']==4
    with pytest.raises(ValueError): components(np.zeros((4,4),dtype=np.uint8))
    with pytest.raises(ValueError): components(np.zeros((4,4),dtype=bool),6)
    with pytest.raises(ValueError): analyze(np.zeros((3,3,3),dtype=np.uint8),np.zeros((4,4)))
