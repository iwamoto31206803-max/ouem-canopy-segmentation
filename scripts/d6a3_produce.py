"""D6 research patch producer, restricted to synthetic A3-1 fixtures."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import shutil

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds

import d6a_research_qa as qa
from m1_5b_pilot import separation
from research_patches import crop_arrays
from r1_dataset import safe_path


PLAN = 'd6-research-plan-v1'
SCHEMA = 'd6-research-manifest-v1'
RECEIPT = 'd6-research-source-v1'
MARKER = b'D6-A3-1 SYNTHETIC FIXTURE\n'
AUTH = {'MATCHED_EXTERNAL_EXPECTED_SHA256', 'SOURCE_AUTHENTICITY_NOT_EXTERNALLY_VERIFIED'}
PRODUCTS = {'rgb.tif': ('uint8', 3), 'canopy_fraction.tif': ('float32', 1), 'canopy_mask.tif': ('uint8', 1)}
REPO = Path(__file__).resolve().parents[1]


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def identifier(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pinned(path, sha):
    need(qa.digest(sha), 'INPUT_PIN_MISMATCH')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == sha, 'INPUT_PIN_MISMATCH')
    return qa.strict_json(raw)


def plain(path):
    """Check ancestors even when the final output does not exist yet."""
    path = path.absolute()
    need('..' not in path.parts, 'UNSAFE_PATH')
    for item in [*reversed(path.parents), path]:
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'UNSAFE_REPARSE_PATH')
    return path


def inventory(directory):
    paths = list(directory.iterdir())
    need(len({p.name.casefold() for p in paths}) == len(paths), 'CASEFOLD_COLLISION')
    return {p.name: plain(p) for p in paths}


def tree_bytes(directory):
    return sum(tree_bytes(p) if p.is_dir() else p.stat().st_size for p in inventory(directory).values())


def write_new(path, data, allowance):
    need(len(data) <= allowance, 'RESOURCE_STOP: disk byte cap')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return len(data)


def source_paths(root, row):
    stem = f"{row['oam_id']}_{row['image_id']}"
    return {k: safe_path(root, f'extracted/dataset/holdout/{folder}/{stem}{suffix}')
            for k, folder, suffix in (('rgb', 'images', '.tif'), ('annotation', 'masks', '.png'))}


def preflight(plan_path, plan_sha, root, evidence):
    need(safe_path(root, 'D6_A3_SYNTHETIC.txt').read_bytes() == MARKER, 'SYNTHETIC_ONLY')
    plan = pinned(plain(plan_path), plan_sha)
    need(plan['schema'] == PLAN and plan['mode'] == 'synthetic'
         and plan['windows_per_source'] == {'train': 1, 'val': 1}, 'PLAN_POLICY')
    limits = plan['limits']
    need(set(limits) == {'commit_bytes', 'rss_bytes', 'host_available_bytes', 'seconds',
                         'output_bytes', 'temporary_bytes'}
         and all(type(x) is int and x > 0 for x in limits.values())
         and limits['temporary_bytes'] <= limits['output_bytes'], 'RESOURCE_LIMITS')
    need(plan['allow_authenticity'] and set(plan['allow_authenticity']) <= AUTH, 'AUTH_POLICY')
    base = plan_path.absolute().parent
    def ref(name):
        item = plan[name]
        return pinned(safe_path(base, item['path']), item['sha256'])
    contract, report, protected = ref('a1_contract'), ref('a1_report'), ref('protected')
    need(contract['schema'] == report['policy'] == qa.POLICY
         and contract['source_generation'] == report['source_generation'] == qa.GENERATION
         and report['contract_sha256'] == plan['a1_contract']['sha256']
         and contract['catalog_sha256'] == report['catalog_sha256']
         and contract['seed'] == report['seed'], 'A1_BINDING')
    need(protected.get('schema') == 'd6-protected-index-v1'
         and set(protected.get('scopes', {})) == {'historical_v5', 'historical_val18', 'gt54'}
         and all(qa.digest(v) for v in protected['scopes'].values())
         and {x.get('scope') for x in protected['sources']} == set(protected['scopes']), 'PROTECTED_EVIDENCE_HOLD')
    paths = {k: safe_path(root, v) for k, v in qa.FILES.items()}
    for k, path in paths.items():
        qa.require_hash(path, contract['catalog_sha256'][k])
    rows, groups, objects = qa.load_catalog(paths)
    candidates = {x['image_id']: x for x in contract['candidates']}
    results = {x['image_id']: x for x in report['results']}
    ids = plan['source_ids']
    need(ids and all(type(i) is int for i in ids) and len(ids) == len(set(ids))
         and len(candidates) == len(contract['candidates']) and len(results) == len(report['results']),
         'DUPLICATE_OR_INVALID_ID')
    need(set(candidates) == set(results), 'A1_COHORT_MISMATCH')
    expected_allow = {role: sorted(i for i,x in results.items()
                     if x.get('split') == role and x['final_decision']['status'] == qa.ELIGIBLE)
                     for role in ('train', 'val')}
    need(report['eligible_ids'] == expected_allow, 'A1_ALLOWLIST_MISMATCH')
    sources = []
    used_groups, used_objects = set(), set()
    for i in ids:
        row, candidate, result = rows[i], candidates[i], results[i]
        split, oid = row['split'], row['oam_id']
        need(split in ('train', 'val') and i in expected_allow[split]
             and row['official_split'] == result['official_split'] == 'train'
             and all(result[k] == candidate[k] == row[k] for k in ('oam_id', 'validation_fold'))
             and result['split'] == split, 'A1_SOURCE_NOT_ADMITTED')
        need(type(row['width']) is int and type(row['height']) is int
             and 0 < row['width'] <= 512 and 0 < row['height'] <= 512, 'SYNTHETIC_SIZE_LIMIT')
        safe_path(evidence, oid+'-stac.json')
        try:
            guard = qa.source_preflight(row, candidate, groups, objects, set(), evidence)
        except qa.Decision as exc:
            raise ValueError(exc.result['reason']) from exc
        alias = guard['source_object_sha256']
        need(oid.casefold() not in used_groups and alias not in used_objects, 'DUPLICATE_SCENE')
        used_groups.add(oid.casefold()); used_objects.add(alias)
        # The pinned protected index derives from historical manifests, not inferred from A1.
        scene = groups[oid]
        for p in protected['sources']:
            box = p['bounds_wgs84']
            need(isinstance(p['oam_id'], str) and qa.digest(p['object_sha256'])
                 and len(box) == 4 and all(type(v) in (int,float) and np.isfinite(v) for v in box)
                 and -180 <= box[0] < box[2] <= 180 and -90 <= box[1] < box[3] <= 90, 'PROTECTED_EVIDENCE_HOLD')
            need(p['oam_id'].casefold() != oid.casefold() and p['object_sha256'] != alias, 'PROTECTED_SOURCE_HOLD')
            try:
                separation(dict(row, wgs84_bounds=scene['wgs84_bounds'], center=scene['center']),
                           [dict(p, image_id=-1, official_split='test', validation_fold=-1,
                                 wgs84_bounds=box, center=[(box[0]+box[2])/2, (box[1]+box[3])/2])])
            except ValueError as exc:
                raise ValueError('PROTECTED_PROXIMITY_HOLD') from exc
        crop = result['crop_xywh']
        need(len(crop) == 4 and all(type(x) is int for x in crop)
             and min(crop[:2]) >= 0 and crop[2:] == [224,224]
             and result['selected_black_support_cells'] == 0, 'A1_CROP')
        need(result['stac_sha256'] == guard['stac_sha256']
             and result['source_object_sha256'] == alias, 'A1_RIGHTS_BINDING')
        hashes = result['source_hashes']
        verification = result['source_hash_verification']
        need(set(hashes) == {'rgb','annotation'} and all(qa.digest(h) for h in hashes.values())
             and verification in plan['allow_authenticity']
             and (candidate.get('source_hashes') == hashes if verification == 'MATCHED_EXTERNAL_EXPECTED_SHA256'
                  else 'source_hashes' not in candidate), 'AUTHENTICITY_BINDING')
        need({k: qa.sha(p) for k,p in source_paths(root,row).items()} == hashes, 'INPUT_PIN_MISMATCH')
        key = identifier([qa.GENERATION, i, oid, alias])
        sources.append(dict(schema=RECEIPT, plan_sha256=plan_sha, source_key=key,
                            patch_id=identifier([plan_sha,key,crop]), image_id=i, oam_id=oid,
                            source_object_sha256=alias, split=split, validation_fold=row['validation_fold'],
                            crop_xywh=crop, input_sha256=hashes, stac_sha256=guard['stac_sha256'],
                            source_hash_verification=verification, final_decision=qa.ELIGIBLE,
                            a1_output_shape=result['output_shape'], a1_crop_transform=result['crop_transform'],
                            categories=result['coco_categories']))
    need({s['split'] for s in sources} == {'train','val'}, 'PLAN_SPLITS_REQUIRED')
    return limits, sources, rows


def grid(shape, affine):
    return dict(shape=list(shape), crs='EPSG:3395', affine=list(affine)[:6],
                bounds=list(array_bounds(*shape, affine)))


def emit(stage, arrays, patch, allowance, hook):
    hashes = {}
    for name, (dtype,count) in PRODUCTS.items():
        if name == 'canopy_mask.tif':
            with rasterio.open(stage/'canopy_fraction.tif') as ds:
                array = (ds.read() >= .5).astype(np.uint8)
        else:
            array = arrays[name]
        profile = dict(driver='GTiff', height=224, width=224, count=count, dtype=dtype,
                       crs=patch['crs'], transform=Affine(*patch['affine']), compress='deflate')
        with MemoryFile() as encoded:
            with encoded.open(**profile) as ds:
                ds.write(array)
            size = len(encoded)
            need(size <= allowance, 'RESOURCE_STOP: encoded TIFF exceeds reserved bytes')
            data = encoded.read()
            need(len(data) == size, 'ENCODED_LENGTH_MISMATCH')
            allowance -= write_new(stage/name, data, allowance)
        with rasterio.open(stage/name) as ds:
            need(np.array_equal(ds.read(),array), 'PRODUCT_READBACK_MISMATCH')
        hashes[name] = dict(path=name, size=size, sha256=qa.sha(stage/name))
        hook(name)
    return hashes, allowance


def validate_source(folder, receipt, expected):
    need(all(receipt[k] == v for k,v in expected.items()), 'SOURCE_RECEIPT_BINDING')
    need(set(inventory(folder)) == {'source.json', *PRODUCTS}, 'SOURCE_INVENTORY')
    x,y,w,h = receipt['crop_xywh']
    full, patch = receipt['full_grid'], receipt['patch_grid']
    shape = full['shape']
    need(full['crs'] == 'EPSG:3395' and len(shape) == 2
         and all(type(v) is int and v>0 for v in shape)
         and all(type(v) is int for v in (x,y,w,h)) and min(x,y)>=0 and (w,h)==(224,224)
         and x+w<=shape[1] and y+h<=shape[0]
         and shape == receipt['a1_output_shape'] and patch['affine'] == receipt['a1_crop_transform'],
         'SOURCE_GRID_BINDING')
    affine = Affine(*full['affine'])
    need(np.isfinite(list(affine)).all() and not affine.b and not affine.d and affine.a>0 and affine.e<0
         and full == grid(shape,affine) and patch == grid((224,224),affine*Affine.translation(x,y)),
         'PATCH_GRID_MISMATCH')
    arrays = {}
    need(set(receipt['products']) == PRODUCTS.keys(), 'PRODUCT_INVENTORY')
    for name,(dtype,count) in PRODUCTS.items():
        pin=receipt['products'][name]
        need(pin['path']==name and type(pin['size']) is int, 'PRODUCT_PATH')
        path=safe_path(folder,name)
        need(path.stat().st_size==pin['size'] and qa.sha(path)==pin['sha256'], 'PRODUCT_INTEGRITY_FAIL')
        with rasterio.open(path) as ds:
            need(ds.driver=='GTiff' and ds.shape==(224,224) and ds.dtypes==(dtype,)*count
                 and str(ds.crs)==patch['crs'] and list(ds.transform)[:6]==patch['affine']
                 and list(ds.bounds)==patch['bounds'] and ds.nodata is None and len(ds.files)==1
                 and np.all(ds.read_masks()==255),
                 'PRODUCT_GRID_OR_DTYPE')
            arrays[name]=ds.read()
        need(qa.sha(path)==pin['sha256'], 'PRODUCT_CHANGED')
    fraction=arrays['canopy_fraction.tif']
    need(np.isfinite(fraction).all() and np.all((fraction>=0)&(fraction<=1))
         and np.array_equal(arrays['canopy_mask.tif'],fraction>=.5), 'SAVED_THRESHOLD_MISMATCH')
    return arrays


def stage_source(root, stage, expected, row, allowance, hook=lambda _: None):
    paths=source_paths(root,row)
    need({k:qa.sha(p) for k,p in paths.items()}==expected['input_sha256'], 'INPUT_PIN_MISMATCH')
    rgb, teacher, shape, affine, gsd=qa.decode(row,paths,expected['categories'])
    arrays, transform=crop_arrays(rgb,teacher,shape,affine,expected['crop_xywh'])
    need(list(shape)==expected['a1_output_shape'] and list(transform)[:6]==expected['a1_crop_transform'],
         'A1_GRID_MISMATCH')
    patch=grid((224,224),transform)
    products,left=emit(stage,arrays,patch,allowance,hook)
    need({k:qa.sha(p) for k,p in paths.items()}==expected['input_sha256'], 'INPUT_PIN_MISMATCH')
    receipt=dict(expected,full_grid=grid(shape,affine),patch_grid=patch,ground_gsd=gsd,products=products)
    write_new(stage/'source.json',canonical(receipt),left)
    hook('source.json')


def worker(root, stage, expected, row, allowance):
    import sys
    def no_network(event, _args):
        if event.startswith('socket.') or event in ('subprocess.Popen','os.system'):
            raise PermissionError('Synthetic offline worker')
    sys.addaudithook(no_network)
    stage_source(Path(root),Path(stage),qa.strict_json(expected),qa.strict_json(row),int(allowance))


def produce(plan, pin, root, evidence, output, *, resume=False, guard=None, hook=lambda _: None):
    from d6a3_guard import watch

    root,evidence,output=map(plain,(root,evidence,output))
    need(not any(output.is_relative_to(p) or p.is_relative_to(output)
                 for p in (root,evidence,REPO,plain(plan).parent)), 'PROTECTED_OUTPUT')
    limits,sources,rows=preflight(plan,pin,root,evidence)
    need(output.parent.is_dir() and shutil.disk_usage(output.parent).free>=limits['output_bytes'], 'DISK_HEADROOM')
    if not resume:
        output.mkdir()
    claim=output/'.claim'
    claim.touch(exist_ok=False)
    try:
        binding=output/'plan.json'
        if not resume:
            write_new(binding,plan.read_bytes(),limits['output_bytes'])
            (output/'sources').mkdir()
            (output/'commits').mkdir()
        pinned(safe_path(output,'plan.json'),pin)
        root_names=set(inventory(output))
        need({'.claim','plan.json','sources','commits'} <= root_names <=
             {'.claim','plan.json','sources','commits','.manifest.partial','manifest.json'}, 'GENERATION_INVENTORY')
        target=output/'sources'
        allow={s['source_key'] for s in sources}
        need(all(n in allow or n in {'.stage-'+k for k in allow} for n in inventory(target)), 'UNEXPECTED_SOURCE_PATH')
        need(set(inventory(output/'commits')) <= {k+'.json' for k in allow}, 'UNEXPECTED_COMMIT_PATH')
        receipts=[]; telemetry=[]; skipped=0
        for source in sources:
            key=source['source_key']; final=target/key; stage=target/('.stage-'+key)
            commit=output/'commits'/(key+'.json')
            need('manifest.json' not in root_names or final.is_dir(), 'FROZEN_SOURCE_MISSING')
            need(not (final.exists() and stage.exists()), 'AMBIGUOUS_SOURCE_STATE')
            need(not commit.exists() or final.exists() or stage.exists(), 'COMMITTED_SOURCE_MISSING')
            if final.exists():
                need(commit.is_file(), 'SOURCE_COMMIT_PIN_MISSING')
                skipped+=1
                folder=final
            else:
                folder=stage
                if stage.exists():
                    need(commit.is_file(), 'UNVERIFIED_STAGE_HOLD')
                else:
                    remaining=min(limits['temporary_bytes'],limits['output_bytes']-tree_bytes(output))
                    need(remaining>0,'RESOURCE_STOP: no reserved bytes')
                    stage.mkdir()
                    args=(root,stage,canonical(source).decode(),canonical(rows[source['image_id']]).decode(),remaining)
                    telemetry.append((guard or watch)(args,limits))
            raw=(folder/'source.json').read_bytes()
            validate_source(folder,qa.strict_json(raw),source)
            need(tree_bytes(output)<=limits['output_bytes']
                 and (folder==final or tree_bytes(stage)<=limits['temporary_bytes']),'RESOURCE_STOP: disk limit')
            ref=dict(source_key=key,sha256=hashlib.sha256(raw).hexdigest())
            committed=canonical(dict(ref,plan_sha256=pin))
            if commit.exists():
                need(commit.read_bytes()==committed, 'SOURCE_COMMIT_PIN_MISMATCH')
            else:
                write_new(commit,committed,limits['output_bytes']-tree_bytes(output))
                hook('source_verified')
            if folder==stage:
                need(not final.exists(),'PUBLISHED_SOURCE_EXISTS')
                stage.rename(final)  # Same parent, exclusive writer, Windows non-replace.
                hook('source_published')
            receipts.append(ref)
        manifest=dict(schema=SCHEMA,plan_sha256=pin,source_receipts=receipts,
                      eligible_ids={r:[s['image_id'] for s in sources if s['split']==r] for r in ('train','val')})
        manifest['generation_id']=identifier(manifest)
        payload=canonical(manifest)
        partial=output/'.manifest.partial'
        published=output/'manifest.json'
        if published.exists():
            need(not partial.exists() and published.read_bytes()==payload,'FROZEN_MANIFEST_MISMATCH')
            return dict(manifest_sha256=qa.sha(published),generation_id=manifest['generation_id'],
                        skipped_sources=skipped,worker_measurements=telemetry)
        if partial.exists():
            need(partial.read_bytes()==payload,'INCOMPLETE_MANIFEST_STAGE')
        else:
            write_new(partial,payload,limits['output_bytes']-tree_bytes(output))
        hook('before_manifest')
        os.link(partial,published)
        partial.unlink()
        hook('after_manifest')
        return dict(manifest_sha256=qa.sha(output/'manifest.json'),generation_id=manifest['generation_id'],
                    skipped_sources=skipped,worker_measurements=telemetry)
    finally:
        claim.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','root','evidence','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        result=produce(args.plan,args.plan_sha256,args.root,args.evidence,args.output,resume=args.resume)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        parser.exit(1,f'D6 STOP: {type(exc).__name__}: {exc.args[0] if isinstance(exc,ValueError) else "inspect private stage"}\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
