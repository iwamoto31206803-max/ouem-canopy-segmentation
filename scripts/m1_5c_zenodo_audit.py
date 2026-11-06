"""Read-only Zenodo/HF compatibility audit; never rewrites source or training pairs."""
import argparse
from collections import Counter,defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from m1_2a_data_qa import labels
from m1_2b_overlay_qa import overlay
from m1_2d_geometry_audit import sha,require_hash,intervals
from m1_2d_prepare_pairs import output_grid,reduce_area,binary
from m1_5b_pilot import fresh,verify_sample

REPO=Path(__file__).resolve().parents[1]
TARGETS=[3,661,1972,4235]


def json_lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    if len({r['image_id'] for r in rows})!=len(rows): raise ValueError('Duplicate image_id')
    return rows


def partition(row,official_split):
    if official_split=='test':
        if row['validation_fold']!=-1: raise ValueError('Holdout fold conflict')
        return 'holdout'
    if row['validation_fold'] not in range(5): raise ValueError('Invalid train fold')
    return 'val' if row['validation_fold']==0 else 'train'


def metadata_audit(train,test,hf):
    combined=train+test
    if len({r['image_id'] for r in combined})!=len(combined): raise ValueError('Train/test ID overlap')
    indexed={r['image_id']:r for r in hf}
    groups=defaultdict(set); roles=Counter()
    for official,rows in [('train',train),('test',test)]:
        for row in rows:
            assigned=partition(row,official); roles[assigned]+=1
            groups[row['oam_id']].add((official,assigned,row['validation_fold']))
    if any(len(v)!=1 for v in groups.values()): raise ValueError('Source crosses split/fold/holdout')
    if {r['image_id'] for r in combined}!=set(indexed): raise ValueError('Dataset ID set differs')
    mismatches={key:[r['image_id'] for r in combined if r.get(key)!=indexed[r['image_id']].get(key)]
        for key in ['oam_id','validation_fold','license','height','width','crs','bounds']}
    for rid,expected in [(3,'train'),(661,'train'),(1972,'val'),(4235,'val')]:
        row=next(r for r in train if r['image_id']==rid)
        if partition(row,'train')!=expected: raise ValueError('Protected role changed')
    return dict(counts={'train_metadata':len(train),'test_metadata':len(test),'roles':dict(roles),'oam_ids':len(groups)},
        licenses={name:dict(Counter(str(r.get('license')) for r in rows)) for name,rows in [('train',train),('test',test)]},
        train_explicit_cc_by=sum(r.get('license')=='CC-BY 4.0' for r in train),
        license_unknown_policy='not eligible; never inherit the HF/archive declaration to fill missing row rights',
        metadata_mismatch_ids=mismatches,source_group_fold_conflicts=0,protected_roles_preserved=True)


def coco_annotation_summary(path,targets):
    """Stream one known COCO annotations array, keeping only small per-target summaries."""
    result={i:dict(count=0,ids=[],categories=set(),object_keys=set(),extra_keys=set(),nonempty_extra_count=0,crowd=0) for i in targets}
    decoder=json.JSONDecoder(); buffer=''
    with path.open(encoding='utf-8') as stream:
        while True:
            chunk=stream.read(65536)
            if not chunk: raise ValueError('Missing COCO annotations array')
            buffer+=chunk
            match=re.search(r'"annotations"\s*:\s*\[',buffer)
            if match: buffer=buffer[match.end():]; break
            if len(buffer)>8*1024**2: raise ValueError('Unexpected COCO prefix')
        while True:
            buffer=buffer.lstrip()
            if buffer.startswith(']'): break
            if buffer.startswith(','): buffer=buffer[1:].lstrip()
            try: obj,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk=stream.read(65536)
                if not chunk: raise ValueError('Truncated COCO annotations')
                buffer+=chunk
                if len(buffer)>16*1024**2: raise ValueError('Unexpectedly large COCO object')
                continue
            buffer=buffer[end:]
            if obj['image_id'] in result:
                entry=result[obj['image_id']];entry['count']+=1;entry['ids'].append(obj['id'])
                entry['categories'].add(obj['category_id']);entry['object_keys'].update(obj);entry['crowd']+=bool(obj.get('iscrowd'))
                extra=obj.get('extra',{})
                entry['nonempty_extra_count']+=bool(extra)
                if isinstance(extra,dict):entry['extra_keys'].update(extra)
    for value in result.values():
        value['categories']=sorted(value['categories']);value['object_keys']=sorted(value['object_keys'])
        value['extra_keys']=sorted(value['extra_keys'])
    return result


def pixel_comparison(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(status='FAIL',shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype)
    delta=a.astype(np.int64)-b.astype(np.int64)
    return dict(status='PASS' if np.array_equal(a,b) else 'FAIL',shape=list(a.shape),dtype=str(a.dtype),
        changed_values=int(np.count_nonzero(delta)),changed_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))) if a.ndim==3 else int(np.count_nonzero(delta)),
        delta_min=int(delta.min()),delta_max=int(delta.max()),mean_absolute_error=float(np.abs(delta).mean()),
        decoded_sha256=[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in [a,b]])


def panoptic_ids(array):
    if array.ndim!=3 or array.shape[2]!=3 or array.dtype!=np.uint8: raise ValueError('Expected RGB uint8 panoptic PNG')
    rgb=array.astype(np.uint32)
    return rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]


def available_reference(folders):
    return next((p for p in folders if all((p/n).is_file() for n in
                ['rgb.bin','annotation.bin','segments.json'])),None)


def geometry(path):
    with rasterio.open(path) as src:
        x,y=src.transform*(src.width/2,src.height/2)
        shape,grid,gsd=output_grid(src.crs,src.transform,src.width,src.height)
        return dict(crs=str(src.crs),shape=list(src.shape),affine=list(src.transform)[:6],bounds=list(src.bounds),
            local_ground_gsd=intervals(src.crs,x,y,*src.res),nodata=src.nodata,bands=src.count,dtypes=list(src.dtypes),
            compression=src.profile.get('compress'),alpha=any(v.name=='alpha' for v in src.colorinterp),
            mask_values=np.unique(src.dataset_mask()).tolist(),mask_flags=[[v.name for v in f] for f in src.mask_flag_enums],
            proposed_050_shape=list(shape),proposed_050_affine=list(grid)[:6],proposed_050_gsd=gsd)


def audit(zenodo,cache_root,inventory_cache,output):
    reference=json.loads((REPO/'docs/implementation/M1_5B_PILOT_RESULT.json').read_text())
    inventory=json.loads((REPO/'docs/implementation/M1_5A_INVENTORY_RESULT.json').read_text())
    require_hash(inventory_cache/'rows.json',inventory['metadata_cache_sha256'])
    hf_rows=json.loads((inventory_cache/'rows.json').read_text()); hf_index={r['image_id']:r for r in hf_rows}
    train=json_lines(zenodo/'archives/train_meta.json'); test=json_lines(zenodo/'archives/test_meta.json')
    meta={r['image_id']:r for r in train}; root=zenodo/'extracted/dataset/holdout'
    summary=metadata_audit(train,test,hf_rows)
    files={'rgb':len(list((root/'images').glob('*.tif'))),'mask':len(list((root/'masks').glob('*.png')))}
    missing=[r['image_id'] for r in train if not all((root/k/(r['oam_id']+'_'+str(r['image_id'])+suffix)).is_file()
        for k,suffix in [('images','.tif'),('masks','.png')])]
    coco=coco_annotation_summary(zenodo/'archives/dataset/holdout/train.json',TARGETS)
    fresh(output,[zenodo,cache_root,inventory_cache])
    source_manifest=json.loads((REPO/'docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())
    pair_manifest=json.loads((REPO/'docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    hf_sources={3:[cache_root/'m1-5b3-pilot-01/raw/3'],
                661:[cache_root/'m1-2a-audit-02/661',cache_root/'m1-2a-audit-01/661'],
                1972:[cache_root/'m1-2a-audit-02/1972']}
    existing={s['image_id']:s for s in source_manifest['samples']}
    receipt=reference['m1_5b3']['external_record_hashes']
    receipt_checks={}
    for relative,key in [('m1-5b3-pilot-01/raw_manifest.json','raw_manifest.json'),
                         ('m1-5b3-pairs-01/manifest.json','manifest.json')]:
        path=cache_root/relative
        if path.is_file():
            require_hash(path,receipt[key]);receipt_checks[relative]='PASS'
        else: receipt_checks[relative]='NOT COMPARABLE: absent'
    protected3=next(s for s in reference['m1_5b3']['samples'] if s['image_id']==3)
    result=dict(scope='Read-only compatibility audit, no source/grid repair or training-pair generation',
        versions={p:version(p) for p in ['numpy','Pillow','rasterio','pyproj']},
        hf_revision=inventory['revision'],zenodo_record='https://zenodo.org/records/11617167',
        archive_md5_status='USER_REPORTED_PASS; archive payloads not rehashed in this representative audit',
        input_metadata_hashes={p.name:sha(p) for p in [zenodo/'archives/train_meta.json',zenodo/'archives/test_meta.json']},
        corpus_metadata=summary,file_counts=files,missing_train_pairs=missing,
        prior_receipt_checks=receipt_checks,samples=[])
    for rid in TARGETS:
        row=meta[rid]; stem=row['oam_id']+'_'+str(rid)
        rgb_path=root/'images'/(stem+'.tif'); mask_path=root/'masks'/(stem+'.png')
        rgb=np.array(Image.open(rgb_path)); ann=np.array(Image.open(mask_path)); ids=panoptic_ids(ann)
        if rid not in meta or partition(row,'train')=='holdout': raise ValueError('Target is holdout')
        geo=geometry(rgb_path); black=np.all(rgb==0,axis=2); edge=np.zeros(black.shape,dtype=bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
        entry=dict(image_id=rid,oam_id=row['oam_id'],role=partition(row,'train'),
            zenodo_source_sha256={'rgb':sha(rgb_path),'mask':sha(mask_path)},zenodo_geometry=geo,
            metadata={key:row[key] for key in ['image_id','oam_id','validation_fold','license','crs','bounds']},
            target_metadata_identity={key:row.get(key)==hf_index[rid].get(key) for key in ['image_id','oam_id','validation_fold','license','crs','bounds']},
            png_mode=Image.open(mask_path).mode,panoptic_nonzero_id_count=len(np.unique(ids))-int(np.any(ids==0)),
            image_mask_shape_equal=rgb.shape[:2]==ann.shape[:2],raw_black_pixels=int(black.sum()),edge_black_pixels=int((black&edge).sum()),
            metadata_dimensions_match=geo['shape']==[row['height'],row['width']],
            orientations=[Image.open(p).getexif().get(274,1) for p in [rgb_path,mask_path]],
            license_gate='PASS' if row.get('license')=='CC-BY 4.0' else 'HOLD_UNRESOLVED_LICENSE',
            coverage_review='REVIEW_REQUIRED' if np.any(black&edge) else 'NO_EDGE_BLACK_TRIGGER',
            coco_annotation_summary=coco[rid],canonical_category_mapping='NOT_VERIFIED_NO_PANOPTIC_ID_CROSSWALK',
            source_compatibility='NOT COMPARABLE',training_eligible=False)
        entry['panoptic_ids_absent_from_coco_annotation_ids']=sorted(set(np.unique(ids).tolist())-{0}-set(coco[rid]['ids']))
        west,south,_,_=row['bounds']
        lon,lat=rasterio.warp.transform(row['crs'],'EPSG:4326',[west],[south])
        aux=[row.get('lat'),row.get('lon')]
        entry['aux_lat_lon']={'lat':aux[0],'lon':aux[1],
            'matches_swapped_lower_left_xy':bool(all(isinstance(v,(int,float)) for v in aux) and np.allclose(aux,[lon[0],lat[0]],atol=1e-5,rtol=0)),
            'used_for_geometry':False}
        expected_grid=from_bounds(*row['bounds'],row['width'],row['height'])
        entry['zenodo_grid_equals_float32_row_bounds']=list(expected_grid)[:6]==geo['affine']
        old=available_reference(hf_sources.get(rid,[]))
        if old:
            expected=protected3['source_hashes'] if rid==3 else existing[rid]['hashes']
            for name in ['rgb.bin','annotation.bin','segments.json']:require_hash(old/name,expected[name])
            original_rgb=np.array(Image.open(old/'rgb.bin')); original_ann=np.array(Image.open(old/'annotation.bin'))
            original_geo=geometry(old/'rgb.bin')
            entry.update(hf_source_sha256={name:sha(old/name) for name in ['rgb.bin','annotation.bin','segments.json']},
                rgb_pixel_comparison=pixel_comparison(rgb,original_rgb),
                panoptic_rgb_comparison=pixel_comparison(ann,original_ann),
                panoptic_id_comparison=pixel_comparison(ids,panoptic_ids(original_ann)),hf_geometry=original_geo,
                geometry_equality={key:geo[key]==original_geo[key] for key in ['crs','shape','affine','bounds']},
                bounds_delta_projected_m=(np.array(geo['bounds'])-original_geo['bounds']).tolist())
            segments=json.loads((old/'segments.json').read_text())
            canonical,qa=labels(rgb,ann,segments); original_binary,_=labels(original_rgb,original_ann,segments)
            entry.update(canonical_category_mapping='PASS_WITH_HASH_VERIFIED_PRIOR_SEGMENTS',panoptic_qa=qa,
                         binary_comparison=pixel_comparison(canonical,original_binary))
            old_pair=protected3 if rid==3 else next(s for s in pair_manifest['samples'] if s['image_id']==rid)
            old_root=cache_root/('m1-5b3-pairs-01' if rid==3 else 'm1-2d-ground-01')
            if all((old_root/old_pair['split']/str(rid)/name).is_file() for name in old_pair['output_hashes']):
                verify_sample(old_root,old_pair)
                shape=tuple(geo['proposed_050_shape']); fraction=np.clip(reduce_area(canonical,shape),0,1).astype(np.float32)
                with rasterio.open(old_root/old_pair['split']/str(rid)/'canopy_mask.tif') as saved:
                    entry['in_memory_zenodo_binary_vs_accepted']=pixel_comparison(binary(fraction),saved.read(1))
                entry['accepted_050_shape']=old_pair['output_shape']
            else: entry['in_memory_zenodo_binary_vs_accepted']={'status':'NOT COMPARABLE','reason':'Accepted products missing'}
            entry['source_compatibility']='PASS' if all(entry[k]['status']=='PASS' for k in
                ['rgb_pixel_comparison','panoptic_rgb_comparison','panoptic_id_comparison','binary_comparison']) and all(entry['geometry_equality'].values()) and all(entry['target_metadata_identity'].values()) and entry['metadata_dimensions_match'] else 'FAIL'
            preview=overlay(rgb,canonical)
        else:
            # No category guess from nonzero IDs. Display ID coverage only, never a semantic target.
            preview=rgb.copy();preview[ids!=0]=np.rint(.65*rgb[ids!=0]+.35*np.array([255,0,255])).astype(np.uint8)
            entry['comparison_reason']='No HF raw RGB/PNG/segments found in prioritized caches; COCO instance IDs are not a proven Panoptic ID-category crosswalk'
        with rasterio.open(rgb_path) as src:
            entry['zenodo_pil_rasterio_equal']=np.array_equal(rgb,src.read().transpose(1,2,0))
        canvas=Image.new('RGB',(rgb.shape[1]*2,rgb.shape[0]));canvas.paste(Image.fromarray(rgb));canvas.paste(Image.fromarray(preview),(rgb.shape[1],0));canvas.thumbnail((1400,700));canvas.save(output/f'{rid}-preview.jpg',quality=90)
        entry['preview_sha256']=sha(output/f'{rid}-preview.jpg');result['samples'].append(entry)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zenodo',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--inventory-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=audit(a.zenodo,a.cache_root,a.inventory_cache,a.output)
    print(json.dumps({s['image_id']:s['source_compatibility'] for s in r['samples']}))
