"""Offline Zenodo v1 adapter; source-bound external reviews required, no network."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import rasterio
import rasterio.warp

from m1_5c_zenodo_audit import json_lines, partition, panoptic_ids, coco_annotation_summary
from m1_5b_pilot import fresh, rights, separation, verify_sample
from m1_2b_overlay_qa import overlay
from m1_2d_geometry_audit import sha, require_hash, intervals
from m1_2d_prepare_pairs import output_grid, reduce_area, binary

GENERATION = 'zenodo-11617167-v1.0.0'


def local_index(root):
    rows, groups, ids = [], {}, set()
    for official in ['train', 'test']:
        for original in json_lines(root/'archives'/f'{official}_meta.json'):
            row = dict(original, official_split=official)
            rid, group = row['image_id'], row['oam_id']
            if type(rid) is not int or rid in ids: raise ValueError('Duplicate/invalid image ID')
            if not isinstance(group,str) or not re.fullmatch(r'[A-Za-z0-9-]+',group): raise ValueError('Unsafe source ID')
            if type(row['validation_fold']) is not int: raise ValueError('Invalid fold type')
            assigned = partition(row,official)
            signature = (official,assigned,row['validation_fold'])
            if group in groups and groups[group] != signature: raise ValueError('Source crosses split/fold/holdout')
            groups[group] = signature; ids.add(rid)
            row['wgs84_bounds'] = list(rasterio.warp.transform_bounds(row['crs'],4326,*row['bounds']))
            b = row['wgs84_bounds']; row['center'] = [(b[0]+b[2])/2,(b[1]+b[3])/2]
            rows.append(row)
    for rid, assigned in [(3,'train'),(661,'train'),(1972,'val')]:
        for row in rows:
            if row['image_id'] == rid and partition(row,row['official_split']) != assigned:
                raise ValueError('Protected role changed')
    return rows


def semantic_binary(rgb, ann, categories):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError('Expected RGB uint8')
    panoptic_ids(ann)  # encoding only; never claim category crosswalk verification
    if ann.shape != rgb.shape: raise ValueError('PNG/TIFF dimensions differ')
    if any(type(c) is not int or c not in (1,2) for c in categories): raise ValueError('Unknown COCO category')
    mask = np.any(ann != 0,axis=2).astype(np.uint8)
    if mask.any() and not categories: raise ValueError('Positive mask without COCO category evidence')
    return mask


def validity(rgb, review):
    black = np.all(rgb == 0,axis=2)
    edge = np.zeros(black.shape,bool); edge[[0,-1],:] = True; edge[:,[0,-1]] = True
    if any(review.get(k) != v for k,v in [('coverage','ALL_VALID'),('alignment','PASS'),('quality','PASS')]):
        raise ValueError('HOLD_VISUAL_OR_VALIDITY_EVIDENCE')
    if review.get('black_wedge') is not False or not review.get('basis'): raise ValueError('HOLD_UNRESOLVED_COVERAGE')
    if black.any() and review.get('black_pixels_reviewed') is not True: raise ValueError('HOLD_BLACK_PIXELS_REVIEW')
    return dict(exact_black_pixels=int(black.sum()),edge_black_pixels=int((black&edge).sum()),
                validity_basis=review['basis'],validity='raster checks plus source-bound all-valid visual review')


def convert(root, row, proof, categories, evidence_dir, output):
    rid = row['image_id']
    if rid == 4235: raise ValueError('HOLD_KNOWN_BLACK_WEDGE_4235')
    if row['license'] != 'CC-BY 4.0': raise ValueError('HOLD_LICENSE')
    stem = row['oam_id']+'_'+str(rid)
    paths = [root/'extracted/dataset/holdout'/d/(stem+suffix) for d,suffix in [('images','.tif'),('masks','.png')]]
    hashes = {p.name:require_hash(p,proof['source_hashes'][p.name]) for p in paths}
    stac_path = Path(proof['stac_path'])
    if not stac_path.is_absolute(): stac_path = evidence_dir/stac_path
    if stac_path.name != row['oam_id']+'-stac.json': raise ValueError('STAC filename mismatch')
    require_hash(stac_path,proof['source_evidence']['snapshot_sha256'])
    stac, rights_info = rights(dict(row,source_evidence=proof['source_evidence']),stac_path.parent)
    with Image.open(paths[0]) as im, Image.open(paths[1]) as ann:
        if im.format != 'TIFF' or ann.format != 'PNG' or im.mode != 'RGB' or ann.mode != 'RGB': raise ValueError('Expected RGB TIFF/PNG')
        if any(x.getexif().get(274,1) != 1 for x in [im,ann]): raise ValueError('Unsupported orientation')
        rgb, annotation = np.array(im), np.array(ann)
    mask = semantic_binary(rgb,annotation,categories)
    info = validity(rgb,proof['review'])
    with rasterio.open(paths[0]) as src:
        t = src.transform
        if str(src.crs) != 'EPSG:3395' or src.shape != (row['height'],row['width']) or src.count != 3: raise ValueError('CRS/dimensions mismatch')
        if t.b or t.d or t.a <= 0 or t.e >= 0 or not np.isfinite(list(t)).all(): raise ValueError('Unsupported affine')
        if not np.allclose(src.bounds,row['bounds'],rtol=0,atol=1e-8): raise ValueError('Metadata bounds mismatch')
        if src.nodata is not None or not np.all(src.dataset_mask() == 255) or any(c.name == 'alpha' for c in src.colorinterp): raise ValueError('HOLD_NODATA_OR_MASK')
        if not np.array_equal(src.read().transpose(1,2,0),rgb): raise ValueError('Raster/PIL mismatch')
        geographic = rasterio.warp.transform_bounds(src.crs,4326,*src.bounds); b = stac['bbox']
        if not (b[0] <= geographic[0] <= geographic[2] <= b[2] and b[1] <= geographic[1] <= geographic[3] <= b[3]): raise ValueError('Outside STAC bbox')
        shape, transform, gsd = output_grid(src.crs,t,src.width,src.height)
        info.update(input_shape=list(src.shape),source_crs=str(src.crs),source_transform=list(t)[:6],source_bounds=list(src.bounds),
            output_crs=str(src.crs),output_bounds=list(src.bounds),output_transform=list(transform)[:6],output_shape=list(shape),
            source_ground_gsd=intervals(src.crs,*(t*(src.width/2,src.height/2)),t.a,-t.e),output_ground_gsd=gsd,
            gsd_error_m={k:gsd[k]-.5 for k in ['east_m','north_m']})
    fraction = np.clip(reduce_area(mask,shape),0,1).astype(np.float32)
    rgb_out = np.floor(np.clip(reduce_area(rgb,shape),0,255)+.5).astype(np.uint8)
    valid = reduce_area(np.ones(mask.shape,np.uint8),shape).astype(np.float32)
    products = {'rgb.tif':rgb_out.transpose(2,0,1),'canopy_fraction.tif':fraction[None],
                'canopy_mask.tif':binary(fraction)[None],'valid_fraction.tif':valid[None]}
    folder = output/partition(row,'train')/str(rid); folder.mkdir(parents=True)
    output_hashes = {}
    for name,array in products.items():
        path = folder/name
        with rasterio.open(path,'w',driver='GTiff',height=shape[0],width=shape[1],count=array.shape[0],dtype=array.dtype,
                           crs=info['output_crs'],transform=transform,compress='deflate') as dst: dst.write(array)
        output_hashes[name] = sha(path)
    info.update(image_id=rid,split=partition(row,'train'),output_hashes=output_hashes)
    verify_sample(output,info,products)
    Image.fromarray(overlay(rgb_out,binary(fraction))).save(folder/'preview.png')
    info.update(source_hashes=hashes,rights=rights_info,canopy_fraction_mean=float(fraction.mean()),binary_canopy_ratio=float(binary(fraction).mean()),
        source_positive_pixels=int(mask.sum()),uniform_background=bool(np.all(mask==0)),uniform_canopy=bool(np.all(mask==1)),
        semantic_policy='upstream_nonzero_binary_with_review',category_crosswalk_status='not_available',category_level_per_id_validation='NOT_VERIFIED',
        coco_categories=categories,crowd_policy='canonical binary retained; cross-format disagreement alone WARN',threshold=.5,
        rgb_method='exact overlap area float64; floor(value+0.5) uint8',fraction_method='same area weights; float32',
        pixel_definition='outer edges preserved; centers col+0.5,row+0.5',preview_sha256=sha(folder/'preview.png'))
    return info


def prepare(root, evidence_path, ids, output):
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if evidence.get('source_generation') != GENERATION: raise ValueError('Source generation mismatch')
    if not evidence.get('archive_evidence'): raise ValueError('Archive authentication evidence required')
    if not ids or len(ids) != len(set(ids)): raise ValueError('Explicit unique IDs required')
    for name in ['train_meta.json','test_meta.json']: require_hash(root/'archives'/name,evidence['metadata_hashes'][name])
    coco_path = root/'archives/dataset/holdout/train.json'; require_hash(coco_path,evidence['coco_sha256'])
    rows = local_index(root); indexed = {r['image_id']:r for r in rows}
    if any(i not in indexed or indexed[i]['official_split'] != 'train' for i in ids): raise ValueError('Holdout/unknown target forbidden')
    coco = coco_annotation_summary(coco_path,ids)
    fresh(output,[root,evidence_path.parent])
    result = dict(source_generation=GENERATION,zenodo_record='https://zenodo.org/records/11617167',
        official_code_revision='05dab754a3fd15b83d6b7606de7310155f4e2df2',
        utc=datetime.now(timezone.utc).isoformat(),evidence_sha256=sha(evidence_path),metadata_hashes=evidence['metadata_hashes'],
        coco_sha256=evidence['coco_sha256'],archive_evidence=evidence.get('archive_evidence'),target_ground_gsd_m=.5,
        versions={p:version(p) for p in ['numpy','Pillow','rasterio','pyproj']},samples=[])
    for rid in ids:
        row = indexed[rid]
        entry = dict(image_id=rid,oam_id=row['oam_id'],split=partition(row,'train'),validation_fold=row['validation_fold'],
                     source_generation=GENERATION,status='HOLD',training_eligible=False)
        result['samples'].append(entry)
        try:
            entry['separation'] = separation(row,rows)
            entry.update(convert(root,row,evidence['samples'][str(rid)],coco[rid]['categories'],evidence_path.parent,output))
            entry.update(status='PASS_CANDIDATE',training_eligible=True)
        except (ValueError,KeyError,OSError) as exc: entry['reason'] = str(exc)
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    result['status'] = 'PASS_CANDIDATE' if all(s['training_eligible'] for s in result['samples']) else 'HOLD'
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zenodo',type=Path,required=True); parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--ids',type=int,nargs='+',default=[3,661,1972])
    args = parser.parse_args()
    result = prepare(args.zenodo,args.evidence,args.ids,args.output)
    print(json.dumps({s['image_id']:s['status'] for s in result['samples']}))
    raise SystemExit(0 if result['status'] == 'PASS_CANDIDATE' else 1)
