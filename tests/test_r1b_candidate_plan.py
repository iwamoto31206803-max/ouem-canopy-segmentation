"""Metadata-only planner: no network or source pixels required."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import r1b_candidate_plan as p


def rows():
    result=[]
    for i in range(40):
        x=-100+i*4
        result.append(dict(image_id=i,oam_id=f'group-{i}',official_split='train',validation_fold=0 if i%5==4 else 1,
            license='CC-BY 4.0',width=2048,height=2048,crs='EPSG:3395',biome=i%4,
            center=[x,10],wgs84_bounds=[x-.001,9.999,x+.001,10.001],meta=f'https://example.org/scene/{i}_meta.json'))
    return result


def test_reproducible_pool_prefix_diversity_and_no_acceptance():
    original=rows();a=p.select(original,{},set(),pool_size=30);b=p.select(list(reversed(original)),{},set(),pool_size=30)
    assert a==b
    c=p.select(original,{},set(),pool_size=35)
    assert a['initial_batch_ids']==c['initial_batch_ids']
    assert len(a['initial_batch_ids'])==25 and a['counts']=={'train':24,'val':6}
    assert all(r['status']=='HOLD' and r['rights']['status']=='HOLD' for r in a['pool'])
    assert all(r['row_license']=='CC-BY 4.0' and r['official_split']=='train' for r in a['pool'])
    assert all(r['canopy_fraction']=='NOT_MEASURED' for r in a['pool'])
    assert p.select(original,{},set(),seed=1,pool_size=30)['initial_batch_ids']!=a['initial_batch_ids']
    assert len({r['oam_id'] for r in a['pool']})==30
    assert original==rows()


@pytest.mark.parametrize('problem',['license','missing','fold','holdout','oam','object','spatial','near','legacy','known_hold'])
def test_exclusions_and_split_controls(problem):
    data=rows();protected={};holds=set()
    if problem=='license':data[0]['license']=None
    elif problem=='missing':data[0].pop('meta')
    elif problem=='fold':data[0]['validation_fold']=8
    elif problem=='holdout':data[0]['official_split']='test';data[0]['validation_fold']=-1
    elif problem=='oam':data[0]['oam_id']=data[4]['oam_id']
    elif problem=='object':data[0]['meta']=data[4]['meta']
    elif problem in ('spatial','near'):
        data[0]['wgs84_bounds']=data[4]['wgs84_bounds'] if problem=='spatial' else [-84.002,9.999,-84.001,10.001]
        data[0]['center']=[-84.0015,10]
    elif problem=='legacy':protected={0:'train'}
    else:holds={0}
    plan=p.select(data,protected,holds,pool_size=30)
    assert 0 not in plan['initial_batch_ids']
    assert plan['reasons_by_image']['0']!='SELECTED_PROVISIONAL_HOLD'


def test_duplicate_ids_legacy_role_and_bounds():
    data=rows()
    with pytest.raises(ValueError,match='Duplicate'):p.select(data+[copy.deepcopy(data[0])],{},set())
    with pytest.raises(ValueError,match='protected'):p.select(data,{0:'val'},set())
    for count in (0,201):
        with pytest.raises(ValueError):p.select(data,{},set(),pool_size=count)
    assert p.identity({'meta':'http://one.invalid/same_meta.json'})==p.identity({'meta':'https://two.invalid/same_meta.json'})


def test_rights_and_hold_history_never_inferred(tmp_path):
    assert p.cached_rights(rows()[0],[tmp_path])['reason']=='STAC_EVIDENCE_MISSING'
    assert {1,2,9,100}<=p.held_ids({'samples':[{'image_id':100,'status':'HOLD'}]})
    first=tmp_path/'a';second=tmp_path/'b';first.mkdir();second.mkdir()
    (first/'group-0-stac.json').write_text('{}');(second/'group-0-stac.json').write_text('{"x":1}')
    assert p.cached_rights(rows()[0],[first,second])['reason']=='CONFLICTING_STAC_SNAPSHOTS'
    assert p.cached_rights(rows()[0],[first])['status']=='HOLD'


def test_budget_measured_and_missing_inputs(tmp_path):
    candidate=dict(image_id=0,oam_id='source')
    base=tmp_path/'extracted/dataset/holdout';(base/'images').mkdir(parents=True);(base/'masks').mkdir()
    (base/'images/source_0.tif').write_bytes(b'x'*100);(base/'masks/source_0.png').write_bytes(b'x'*10)
    measured={k:dict(count=1,max_bytes=100,total_bytes=100) for k in ('pair_products','previews','documents','stac')}
    r=p.resources(tmp_path,[candidate],measured,200)
    assert r['image_annotation_network_payload_bytes']==0 and r['http_range_cache_bytes']==0
    assert r['proposed_metadata_transfer_limit_bytes']==2*p.MIB
    assert r['incremental_disk_estimate']['modeled_incremental_peak_bytes']==1110+65536
    assert r['incremental_disk_estimate']['proposed_incremental_storage_limit_bytes']>=2*(1110+65536)
    assert r['incremental_disk_estimate']['safety_factor']==2
    (base/'images/source_0.tif').unlink()
    assert p.resources(tmp_path,[candidate],measured)['estimate_status']=='HOLD_MISSING_MEASUREMENTS'
    assert p.resources(tmp_path,[candidate],measured)['image_annotation_network_payload_bytes']=='NOT_ESTIMATED_STOP'


def test_cli_overwrite_and_source_protection(tmp_path):
    root=tmp_path/'source';root.mkdir();report=tmp_path/'existing.json';report.write_text('keep')
    args=['--zenodo-root',str(root),'--ledger','unused','--g0-manifest','unused','--output',str(report)]
    with pytest.raises(ValueError,match='Fresh'):p.main(args)
    assert report.read_text()=='keep'
    args[-1]=str(root/'new.json')
    with pytest.raises(ValueError,match='protected'):p.main(args)
    assert not (root/'new.json').exists()


def test_metadata_generation_and_hash_pin(tmp_path):
    (tmp_path/'archives').mkdir()
    hashes={}
    for name in ('train_meta.json','test_meta.json'):
        path=tmp_path/'archives'/name;path.write_text('{}\n');hashes[name]=p.sha(path)
    ledger={'m1_5d2':{'result':dict(source_generation=p.GENERATION,metadata_hashes=hashes)}}
    p.verify_metadata(tmp_path,ledger)
    ledger['m1_5d2']['result']['source_generation']='HF'
    with pytest.raises(ValueError,match='generation'):p.verify_metadata(tmp_path,ledger)
    ledger['m1_5d2']['result']['source_generation']=p.GENERATION
    (tmp_path/'archives/train_meta.json').write_text('changed')
    with pytest.raises(ValueError,match='SHA256'):p.verify_metadata(tmp_path,ledger)
