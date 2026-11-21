import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import d6a3_produce as p
import d6a3_guard
from d6a3_data import D6Data
from r2_train import make_plan
from research_patches import crop_arrays


@pytest.fixture
def synthetic(tmp_path):
    inputs=tmp_path/'inputs'
    root,evidence=inputs/'cache',inputs/'stac'
    evidence.mkdir(parents=True)
    all_files={k: root/v for k,v in p.qa.FILES.items()}
    all_files['coco'].parent.mkdir(parents=True)
    (root/'D6_A3_SYNTHETIC.txt').write_bytes(p.MARKER)
    folders={d:root/f'extracted/dataset/holdout/{d}' for d in ('images','masks')}
    for path in folders.values(): path.mkdir(parents=True)
    rows=[]; candidates=[]; results=[]; arrays={}
    for rid,fold,x in ((1,1,1000),(2,0,20000),(3,-1,40000)):
        oid=f'fixture-{rid}'
        transform=from_origin(x,128,.47,.46)
        rgb=np.broadcast_to((np.arange(320)%256)[None,:,None],(320,320,3)).astype(np.uint8).copy()
        teacher=np.full((320,320),int(rid==2),np.uint8)
        if rid==2:
            teacher[:,155:165]=0
        row=dict(image_id=rid,oam_id=oid,validation_fold=fold,license='CC-BY 4.0',
                 width=320,height=320,crs='EPSG:3395',bounds=list(rasterio.transform.array_bounds(320,320,transform)),
                 meta=f'https://metadata.example/{oid}.json')
        rows.append(row)
        if fold==-1:
            continue
        image,ann=folders['images']/f'{oid}_{rid}.tif',folders['masks']/f'{oid}_{rid}.png'
        with rasterio.open(image,'w',driver='GTiff',height=320,width=320,count=3,dtype='uint8',
                           crs=row['crs'],transform=transform) as ds:
            ds.write(rgb.transpose(2,0,1))
        Image.fromarray(np.repeat(teacher[:,:,None],3,axis=2)).save(ann)
        ring=[[-1,-1],[2,-1],[2,1],[-1,1],[-1,-1]]
        stac=evidence/f'{oid}-stac.json'
        stac.write_bytes(p.canonical(dict(type='Feature',stac_version='1.0.0',id=oid,bbox=[-1,-1,2,1],
            geometry=dict(type='Polygon',coordinates=[ring]),
            properties={'license':'cc-by-4.0','oam:producer_name':'Fixture','providers':[{'name':'Fixture','roles':['licensor']}]},
            assets={'metadata':{'href':row['meta']},'visual':{'href':'https://metadata.example/image'}})))
        hashes=dict(rgb=p.qa.sha(image),annotation=p.qa.sha(ann))
        can=dict(image_id=rid,oam_id=oid,validation_fold=fold,stac_sha256=p.qa.sha(stac))
        if rid==1: can['source_hashes']=hashes
        candidates.append(can)
        shape,grid,_=p.qa.output_grid(row['crs'],transform,320,320)
        crop=[37,61,224,224]
        arrays[rid],patch=crop_arrays(rgb,teacher,shape,grid,crop)
        results.append(dict(image_id=rid,oam_id=oid,official_split='train',validation_fold=fold,
             split='train' if fold else 'val',crop_xywh=crop,crop_transform=list(patch)[:6],output_shape=list(shape),
             selected_black_support_cells=0,final_decision=dict(status=p.qa.ELIGIBLE),
             stac_sha256=can['stac_sha256'],source_object_sha256=p.qa.identity(row),
             source_hashes=hashes,source_hash_verification=(
                 'MATCHED_EXTERNAL_EXPECTED_SHA256' if rid==1 else 'SOURCE_AUTHENTICITY_NOT_EXTERNALLY_VERIFIED'),
             coco_categories=[1] if rid==2 else []))
    for name,seq in [('train',rows[:2]),('test',rows[2:])]:
        all_files[name].write_bytes(b''.join(p.canonical(r) for r in seq))
    all_files['coco'].write_bytes(p.canonical({'annotations':[{'image_id':2,'id':2,'category_id':1}]}))
    catalog={k:p.qa.sha(v) for k,v in all_files.items()}
    contract=dict(schema=p.qa.POLICY,source_generation=p.qa.GENERATION,seed=0,catalog_sha256=catalog,candidates=candidates)
    (inputs/'a1.json').write_bytes(p.canonical(contract))
    report=dict(policy=p.qa.POLICY,source_generation=p.qa.GENERATION,seed=0,catalog_sha256=catalog,results=results,
                eligible_ids={'train':[1],'val':[2]},contract_sha256=p.qa.sha(inputs/'a1.json'))
    (inputs/'report.json').write_bytes(p.canonical(report))
    scopes={k:'1'*64 for k in ('historical_v5','historical_val18','gt54')}
    protected=dict(schema='d6-protected-index-v1',scopes=scopes,sources=[
        dict(scope=k,oam_id=k,object_sha256='2'*64,bounds_wgs84=[10,10,11,11]) for k in scopes])
    (inputs/'protected.json').write_bytes(p.canonical(protected))
    plan=dict(mode='synthetic',schema=p.PLAN,windows_per_source={'train':1,'val':1},
              limits=dict(commit_bytes=1024**3,rss_bytes=1024**3,host_available_bytes=1,seconds=60,
                          output_bytes=2*1024**2,temporary_bytes=1024**2),
              allow_authenticity=sorted(p.AUTH),source_ids=[1,2])
    for name,file in [('a1_contract','a1.json'),('a1_report','report.json'),('protected','protected.json')]:
        plan[name]=dict(path=file,sha256=p.qa.sha(inputs/file))
    path=inputs/'plan.json'
    out=tmp_path/'output'
    def run(*,resume=False,guard=None,hook=lambda _:None):
        path.write_bytes(p.canonical(plan))
        return p.produce(path,p.qa.sha(path),root,evidence,out,resume=resume,
                         guard=guard or fake_guard,hook=hook)
    return run,out,plan,arrays,inputs


def fake_guard(args,_limits):
    root,stage,expected,row,remaining=args
    p.stage_source(root,stage,json.loads(expected),json.loads(row),remaining)
    return {'guard':'SYNTHETIC_FAKE'}


def test_reader_full_grid_contract(synthetic):
    run,out,_,arrays,_=synthetic
    result=run()
    data=D6Data(out,result['manifest_sha256'])
    plan=make_plan(data,steps=3)
    assert [x['image_id'] for x in plan['train']]==[1,1,1]
    for role in ('train','val'):
        item=plan[role][0]
        assert item['crop_xywh']==[37,61,224,224]
        rgb,mask,read=data.patch(item,role)
        assert list(rgb.shape)==[1,3,224,224] and list(mask.shape)==[1,224,224]
        for name in p.PRODUCTS: assert np.array_equal(read[name],arrays[item['image_id']][name])
        with pytest.raises(ValueError): data.patch(item,'val' if role=='train' else 'train')
        with pytest.raises(ValueError): data.patch(dict(item,crop_xywh=[0,0,224,224]),role)
        with pytest.raises(ValueError): data.patch(dict(item,oam_id='other'),role)
    assert run(resume=True)['skipped_sources']==2


@pytest.mark.parametrize('where',['rgb.tif','canopy_fraction.tif','canopy_mask.tif','source.json',
                                  'source_verified','source_published','before_manifest','after_manifest'])
def test_interrupt_publication_and_resume(synthetic,where):
    run,out,_,_,_=synthetic
    def fail(name):
        if name==where: raise RuntimeError('injected crash')
    def guard(args,_):
        root,stage,expected,row,remaining=args
        p.stage_source(root,stage,json.loads(expected),json.loads(row),remaining,fail)
        return {}
    with pytest.raises(RuntimeError,match='injected'):
        run(guard=guard,hook=fail)
    if where!='after_manifest':
        assert not (out/'manifest.json').exists()
        with pytest.raises(ValueError): D6Data(out,'0'*64)
    if where.endswith('.tif') or where=='source.json':
        with pytest.raises(ValueError,match='UNVERIFIED_STAGE'): run(resume=True)
        assert not (out/'manifest.json').exists()
    else:
        result=run(resume=True)
        assert len(D6Data(out,result['manifest_sha256']).samples)==2


@pytest.mark.parametrize('mode',['corrupt','missing','orphan_claim','input','casefold'])
def test_fail_closed_without_repair(synthetic,mode):
    run,out,_,_,inputs=synthetic
    result=run()
    source=next((out/'sources').iterdir())/'rgb.tif'
    if mode=='corrupt':
        source.write_bytes(b'corrupt')
    elif mode=='missing':
        source.unlink()
    elif mode=='orphan_claim':
        (out/'.claim').touch()
    elif mode=='input':
        next((inputs/'cache/extracted/dataset/holdout/images').glob('*.tif')).write_bytes(b'tampered')
    else:
        (out/'sources'/'UNEXPECTED').mkdir()
    with pytest.raises((ValueError,OSError)):
        run(resume=True)
    assert p.qa.sha(out/'manifest.json')==result['manifest_sha256']


def test_disk_bound_before_publish(synthetic):
    run,out,plan,_,_=synthetic
    plan['limits']['temporary_bytes']=128
    with pytest.raises(ValueError,match='disk|reserved'):
        run()
    assert not (out/'manifest.json').exists()


def test_source_authenticity_policy(synthetic):
    run,out,plan,_,_=synthetic
    plan['allow_authenticity']=['MATCHED_EXTERNAL_EXPECTED_SHA256']
    with pytest.raises(ValueError,match='AUTHENTICITY'):
        run()
    assert not out.exists()


def test_threshold_from_saved_half(tmp_path):
    arrays={'rgb.tif':np.zeros((3,224,224),np.uint8),
            'canopy_fraction.tif':np.full((1,224,224),.5,np.float32),
            'canopy_mask.tif':np.zeros((1,224,224),np.uint8)}
    p.emit(tmp_path,arrays,p.grid((224,224),from_origin(10,10,.5,.5)),1024**2,lambda _:None)
    with rasterio.open(tmp_path/'canopy_mask.tif') as ds:
        assert np.all(ds.read()==1)


@pytest.mark.parametrize('kind',['protected_alias','protected_overlap','missing_scope','rights','a1_hold',
                                 'fold','test_overlap','duplicate'])
def test_source_contract_gate(synthetic,kind):
    run,out,plan,_,inputs=synthetic
    path=inputs/'protected.json'
    value=json.loads(path.read_bytes())
    report=inputs/'report.json'
    rows=json.loads(report.read_bytes())
    if kind=='protected_alias':
        value['sources'][0]['object_sha256']=rows['results'][0]['source_object_sha256']
    elif kind=='protected_overlap':
        value['sources'][0]['bounds_wgs84']=[0,0,.02,.02]
    elif kind=='missing_scope':
        del value['scopes']['gt54']
    elif kind=='rights':
        (inputs/'stac/fixture-1-stac.json').write_text('{}')
    elif kind=='fold':
        rows['results'][0]['validation_fold']=0
    elif kind=='a1_hold':
        rows['results'][0]['final_decision']['status']=p.qa.HOLD
    elif kind=='test_overlap':
        path2=inputs/'cache'/p.qa.FILES['test']
        official=json.loads(path2.read_bytes())
        official['bounds']=[1000,0,1100,128]
        path2.write_bytes(p.canonical(official))
        path3=inputs/'a1.json'
        contract=json.loads(path3.read_bytes())
        contract['catalog_sha256']['test']=p.qa.sha(path2)
        path3.write_bytes(p.canonical(contract))
        plan['a1_contract']['sha256']=p.qa.sha(path3)
        rows['catalog_sha256']=contract['catalog_sha256']
        rows['contract_sha256']=p.qa.sha(path3)
    else:
        plan['source_ids']=[1,1,2]
    path.write_bytes(p.canonical(value))
    report.write_bytes(p.canonical(rows))
    plan['protected']['sha256']=p.qa.sha(path)
    plan['a1_report']['sha256']=p.qa.sha(report)
    with pytest.raises(ValueError):
        run()
    assert not out.exists()


def test_path_traversal_and_reparse(synthetic,tmp_path,monkeypatch):
    run,out,plan,_,inputs=synthetic
    plan['protected']['path']='../protected.json'
    with pytest.raises(ValueError,match='Unsafe'):
        run()
    assert not out.exists()
    original=Path.lstat
    def fake_reparse(path):
        actual=original(path)
        if path==inputs:
            from types import SimpleNamespace
            return SimpleNamespace(st_mode=actual.st_mode,st_file_attributes=1024)
        return actual
    monkeypatch.setattr(Path,'lstat',fake_reparse)
    with pytest.raises(ValueError,match='REPARSE'):
        p.plain(inputs/'not-created')


@pytest.mark.skipif(sys.platform!='win32', reason='Windows Job integration')
def test_real_job_on_synthetic_sources(synthetic):
    run,out,_,_,_=synthetic
    result=run(guard=d6a3_guard.watch)
    assert all(m['peak_rss_bytes']>0 and m['peak_commit_bytes']>0 for m in result['worker_measurements'])
    assert len(D6Data(out,result['manifest_sha256']).samples)==2


@pytest.mark.skipif(sys.platform!='win32', reason='Windows Job integration')
def test_real_working_set_limit_blocks_publication(synthetic):
    run,out,plan,_,_=synthetic
    plan['limits']['rss_bytes']=1
    with pytest.raises(RuntimeError,match='RESOURCE_STOP'):
        run(guard=d6a3_guard.watch)
    assert not (out/'manifest.json').exists()


@pytest.mark.skipif(sys.platform!='win32', reason='Windows Job integration')
def test_real_timeout_reaps_worker(synthetic, monkeypatch):
    run,out,plan,_,_=synthetic
    plan['limits']['seconds']=1
    original=d6a3_guard.subprocess.Popen
    processes=[]
    def slow(args,**kwargs):
        args[args.index('-c')+1]="import sys,time; sys.stdin.buffer.read(1); time.sleep(30)"
        child=original(args,**kwargs)
        processes.append(child)
        return child
    monkeypatch.setattr(d6a3_guard.subprocess,'Popen',slow)
    with pytest.raises(RuntimeError,match='deadline'):
        run(guard=d6a3_guard.watch)
    assert len(processes)==1 and processes[0].poll() is not None
    assert not (out/'manifest.json').exists()
