"""G0: freeze verified existing Zenodo10 bytes; portable verification, no training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

GENERATION = 'zenodo-11617167-v1.0.0'
ROLES = {3:'train',661:'train',1022:'train',1280:'train',2243:'train',493:'train',1358:'train',
         1972:'val',269:'val',3785:'val'}
PRODUCTS = {'rgb.tif','canopy_fraction.tif','canopy_mask.tif','valid_fraction.tif'}
ACCEPTED = 'accepted_for_M1_7_smoke'
REPO = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = 'c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca'
DATASET_ID = 'ouem-zenodo10-e3db510b1125931d'
MAX_TRANSFER_BYTES = 64*1024**2


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha(path) != expected: raise ValueError('SHA256 mismatch: '+path.name)


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True)+'\n').encode()


def identify(manifest):
    core = {k:v for k,v in manifest.items() if k!='dataset_id'}
    return 'ouem-zenodo10-'+hashlib.sha256(encoded(core)).hexdigest()[:16]


def check_evidence(sample):
    """Never promote merely because a prior run says training_eligible."""
    if sample.get('source_generation') != GENERATION: raise ValueError('Source generation mismatch')
    if sample.get('status') not in ('PASS_CANDIDATE',ACCEPTED): raise ValueError('Prior QA not passed')
    if sample.get('semantic_policy')!='upstream_nonzero_binary_with_review' or sample.get('threshold')!=.5:
        raise ValueError('Unapproved semantic policy')
    if sample.get('category_crosswalk_status')!='not_available' or sample.get('category_level_per_id_validation')!='NOT_VERIFIED':
        raise ValueError('Category claim differs from approved evidence')
    rights=sample.get('rights',{})
    if rights.get('row_license')!='CC-BY 4.0' or rights.get('image_license','').lower()!='cc-by-4.0' or rights.get('annotation_license')!='CC BY 4.0':
        raise ValueError('Missing rights evidence')
    for key in ['stac_sha256','stac_url','attribution','annotation_terms']:
        if not rights.get(key): raise ValueError('Missing rights reference: '+key)
    if not sample.get('validity_basis'): raise ValueError('Missing source-bound visual evidence')


def check_manifest(manifest):
    if manifest.get('source_generation')!=GENERATION or manifest.get('acceptance')!=ACCEPTED:
        raise ValueError('Dataset not accepted Zenodo generation')
    samples=manifest['samples']
    if len(samples)!=10 or {s['image_id']:s['split'] for s in samples}!=ROLES:
        raise ValueError('Exactly the frozen ten IDs/roles required')
    if len({s['oam_id'] for s in samples})!=10: raise ValueError('Duplicate oam_id')
    for s in samples:
        check_evidence(s)
        if s['status']!=ACCEPTED or s.get('official_split')!='train': raise ValueError('Holdout/unaccepted sample')
        if type(s['validation_fold']) is not int or s['validation_fold'] not in range(5): raise ValueError('Invalid fold')
        if ('val' if s['validation_fold']==0 else 'train')!=s['split']: raise ValueError('Fold/role mismatch')
        stem=s['oam_id']+'_'+str(s['image_id'])
        if set(s['source_hashes'])!={stem+'.tif',stem+'.png'} or set(s['output_hashes'])!=PRODUCTS:
            raise ValueError('Incomplete source/product receipt')
        for digest in list(s['source_hashes'].values())+list(s['output_hashes'].values()):
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest): raise ValueError('Invalid SHA256')
        evidence=s.get('acceptance_evidence',{})
        for key in ['ledger_pointer','run_manifest_sha256','review_receipt_sha256','review_basis','review_preview_sha256','source_checks']:
            if not evidence.get(key): raise ValueError('Missing promotion evidence: '+key)
    for a in samples:
        for b in samples:
            if a['split']==b['split']: continue
            x=transform_bounds(a['output_crs'],4326,*a['output_bounds'])
            y=transform_bounds(b['output_crs'],4326,*b['output_bounds'])
            if max(x[0],y[0])<min(x[2],y[2]) and max(x[1],y[1])<min(x[3],y[3]):
                raise ValueError('Train/Val footprint overlap')
    if manifest.get('dataset_id')!=identify(manifest): raise ValueError('Dataset identity mismatch')


def check_products(root, sample):
    arrays={}; folder=root/sample['split']/str(sample['image_id'])
    for name,digest in sample['output_hashes'].items():
        path=folder/name; require_hash(path,digest)
        with rasterio.open(path) as src:
            if str(src.crs)!=sample['output_crs'] or src.transform!=Affine(*sample['output_transform']) or list(src.shape)!=sample['output_shape']:
                raise ValueError('Product grid mismatch')
            if not np.allclose(src.bounds,sample['output_bounds'],rtol=0,atol=1e-8): raise ValueError('Product extent mismatch')
            if src.nodata is not None or not np.all(src.dataset_mask()==255): raise ValueError('Invalid product coverage')
            arrays[name]=src.read()
    rgb=arrays['rgb.tif']; fraction=arrays['canopy_fraction.tif']; mask=arrays['canopy_mask.tif']; valid=arrays['valid_fraction.tif']
    h,w=sample['output_shape']
    if rgb.shape!=(3,h,w) or rgb.dtype!=np.uint8 or mask.shape!=(1,h,w) or mask.dtype!=np.uint8:
        raise ValueError('Product channels/dtype mismatch')
    for array in [fraction,valid]:
        if array.shape!=(1,h,w) or array.dtype!=np.float32 or not np.isfinite(array).all() or not ((array>=0)&(array<=1)).all():
            raise ValueError('Invalid fraction')
    if not np.all(valid==1) or not np.array_equal(mask,(fraction>=.5).astype(np.uint8)):
        raise ValueError('Mask threshold/validity mismatch')


def verify(root, manifest_path, expected_sha256):
    require_hash(manifest_path,expected_sha256)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')); check_manifest(manifest)
    expected=set()
    for sample in manifest['samples']:
        check_products(root,sample)
        expected.update(root/sample['split']/str(sample['image_id'])/name for name in PRODUCTS)
    actual={p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.tif','.tiff')}
    if actual!=expected: raise ValueError('Expected exactly 40 product TIFFs/relative paths')
    return dict(status='PASS',dataset_id=manifest['dataset_id'],manifest_sha256=expected_sha256,
                products=40,train=7,val=3,acceptance=ACCEPTED)


def payload(manifest_bytes):
    if hashlib.sha256(manifest_bytes).hexdigest()!=MANIFEST_SHA256: raise ValueError('Frozen manifest SHA mismatch')
    manifest=json.loads(manifest_bytes);check_manifest(manifest)
    if manifest['dataset_id']!=DATASET_ID: raise ValueError('Frozen dataset ID mismatch')
    files={'manifest.json':MANIFEST_SHA256}
    for s in manifest['samples']:
        for name,digest in s['output_hashes'].items(): files[f"{s['split']}/{s['image_id']}/{name}"]=digest
    if len(files)!=41: raise ValueError('Expected 41 transfer members')
    return files


def fresh_external(path, protected):
    resolved=path.resolve()
    if path.exists() or path.is_symlink() or resolved==REPO or REPO in resolved.parents:
        raise ValueError('Use a fresh external destination')
    for source in protected:
        source=source.resolve()
        if resolved==source or source in resolved.parents or resolved in source.parents:
            raise ValueError('Destination overlaps protected input')
    path.mkdir(parents=True,exist_ok=False)


def pack(root, output):
    # Verify every source BEFORE creating any output. No source file is modified.
    validation=verify(root,root/'manifest.json',MANIFEST_SHA256)
    members=payload((root/'manifest.json').read_bytes())
    fresh_external(output,[root])
    archive=output/'zenodo10-g0.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=False) as zipped:
        for name,digest in sorted(members.items()):
            source=root/name;require_hash(source,digest)
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16
            with zipped.open(info,'w') as destination,source.open('rb') as stream:
                shutil.copyfileobj(stream,destination,1024**2)
            require_hash(source,digest)
    inventory=inspect_archive(archive,sha(archive))
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),dataset=validation,
                zip_name=archive.name,zip_sha256=sha(archive),zip_bytes=archive.stat().st_size,
                compression='ZIP_STORED; fixed member order/time/mode',file_count=len(inventory),files=inventory)
    (output/'transfer.json').write_bytes(encoded(result))
    return result


def inspect_archive(archive, expected_zip_sha256):
    """Validate all member names, types, sizes and bytes before extraction creates a directory."""
    if archive.stat().st_size>MAX_TRANSFER_BYTES: raise ValueError('ZIP size limit exceeded')
    require_hash(archive,expected_zip_sha256)
    with zipfile.ZipFile(archive) as zipped:
        infos=zipped.infolist();seen=set();total=0
        if len(infos)!=41: raise ValueError('Expected exactly 41 archive members')
        for info in infos:
            name=info.filename
            if info.orig_filename!=name or '\\' in name or ':' in name or name.startswith('/') or any(p in ('','.','..') for p in name.split('/')):
                raise ValueError('Unsafe archive path')
            if name.casefold() in seen: raise ValueError('Duplicate archive path')
            seen.add(name.casefold())
            mode=info.external_attr>>16
            if info.is_dir() or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or info.flag_bits&1:
                raise ValueError('Non-regular/encrypted archive member')
            if info.compress_type!=zipfile.ZIP_STORED: raise ValueError('Unexpected compression method')
            total+=info.file_size
            if info.file_size<0 or total>MAX_TRANSFER_BYTES: raise ValueError('Expanded size limit exceeded')
        # Only this exact, trusted G0 manifest determines the whitelist.
        try: manifest_bytes=zipped.read('manifest.json')
        except KeyError: raise ValueError('Missing transfer manifest') from None
        expected=payload(manifest_bytes)
        if {i.filename for i in infos}!=set(expected): raise ValueError('Missing/extra transfer member')
        result=[]
        for info in infos:
            digest=hashlib.sha256();count=0
            with zipped.open(info) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):
                    count+=len(chunk)
                    if count>info.file_size or count>MAX_TRANSFER_BYTES: raise ValueError('Member size mismatch')
                    digest.update(chunk)
            if count!=info.file_size or digest.hexdigest()!=expected[info.filename]: raise ValueError('Member SHA/size mismatch')
            result.append(dict(path=info.filename,bytes=count,sha256=digest.hexdigest()))
    return sorted(result,key=lambda item:item['path'])


def unpack(archive, expected_zip_sha256, output):
    inventory=inspect_archive(archive,expected_zip_sha256)
    fresh_external(output,[archive])
    with zipfile.ZipFile(archive) as zipped:
        for item in inventory:
            destination=output/item['path'];destination.parent.mkdir(parents=True,exist_ok=True)
            # No extractall: names are canonical whitelist members and every file is exclusive-create.
            with zipped.open(item['path']) as source,destination.open('xb') as stream:
                shutil.copyfileobj(source,stream,1024**2)
            require_hash(destination,item['sha256'])
    require_hash(archive,expected_zip_sha256)
    validation=verify(output,output/'manifest.json',MANIFEST_SHA256)
    return dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),zip_sha256=expected_zip_sha256,
                zip_bytes=archive.stat().st_size,file_count=len(inventory),dataset=validation,
                checks=['ZIP_SHA','CANONICAL_WHITELIST','NO_DUPLICATES_OR_LINKS','ALL_MEMBER_SHA',
                        'FIXED_MANIFEST_SHA','40_PRODUCTS_GRID_DTYPE_LABEL_VALIDITY','TRAIN7_VAL3'])


def unpack_report(archive, expected_zip_sha256, output, result_json):
    # Reserve the new report before touching output. Never replace an earlier result.
    if output.resolve()==result_json.resolve() or output.resolve() in result_json.resolve().parents:
        raise ValueError('Keep result JSON outside the frozen dataset')
    with result_json.open('xb') as report:
        try: result=unpack(archive,expected_zip_sha256,output)
        except Exception as exc:
            result=dict(status='FAIL',utc=datetime.now(timezone.utc).isoformat(),error_type=type(exc).__name__,
                        error=str(exc),expected_zip_sha256=expected_zip_sha256,
                        dataset_usable=False,note='Do not use partial output; diagnose and retry with fresh paths')
        report.write(encoded(result))
    return result


def freeze(zenodo,d2_root,d3_root,d2_evidence,d3_evidence,output):
    # Heavy historical/source dependencies are intentionally stage-only, not needed on Colab verify.
    from PIL import Image
    from m1_5d_zenodo_prepare import local_index,semantic_binary,validity
    from m1_5b_pilot import fresh,rights,separation
    ledger_path=REPO/'docs/implementation/M1_5B_PILOT_RESULT.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    rows=local_index(zenodo); indexed={r['image_id']:r for r in rows}
    samples=[]; source_roots={}
    for stage,root,proof_path,key in [('m1_5d2',d2_root,d2_evidence,'repeat'),('m1_5d3',d3_root,d3_evidence,'pairs-repeat-01')]:
        record=ledger[stage]; run=record['result']
        expected=record['run_manifest_sha256' if stage=='m1_5d2' else 'manifests_sha256'][key]
        require_hash(root/'manifest.json',expected)
        if json.loads((root/'manifest.json').read_text())!=run: raise ValueError('Run manifest/ledger mismatch')
        require_hash(proof_path,run['evidence_sha256']); proof=json.loads(proof_path.read_text())
        if proof['source_generation']!=GENERATION or run['source_generation']!=GENERATION or not proof.get('archive_evidence'):
            raise ValueError('Missing Zenodo source/authentication chain')
        for name,digest in run['metadata_hashes'].items():
            if proof['metadata_hashes'][name]!=digest: raise ValueError('Metadata receipt mismatch')
            require_hash(zenodo/'archives'/name,digest)
        require_hash(zenodo/'archives/dataset/holdout/train.json',run['coco_sha256'])
        if proof['coco_sha256']!=run['coco_sha256']: raise ValueError('COCO receipt mismatch')
        for original in run['samples']:
            s=json.loads(json.dumps(original)); check_evidence(s)
            rid=s['image_id']; row=indexed[rid]; p=proof['samples'][str(rid)]
            if rid not in ROLES or row['official_split']!='train' or any(s[k]!=row[k] for k in ['oam_id','validation_fold']):
                raise ValueError('Source identity/split mismatch')
            leak=separation(row,rows)
            if s['source_hashes']!=p['source_hashes']: raise ValueError('Source/review binding mismatch')
            for filename,digest in s['source_hashes'].items():
                directory='images' if filename.endswith('.tif') else 'masks'
                require_hash(zenodo/'extracted/dataset/holdout'/directory/filename,digest)
            stac=Path(p['stac_path'])
            if not stac.is_absolute(): stac=proof_path.parent/stac
            require_hash(stac,s['rights']['stac_sha256'])
            _,verified_rights=rights(dict(row,source_evidence=p['source_evidence']),stac.parent)
            if verified_rights!=s['rights']: raise ValueError('STAC rights receipt differs')
            stem=row['oam_id']+'_'+str(rid); base=zenodo/'extracted/dataset/holdout'
            with Image.open(base/'images'/(stem+'.tif')) as im,Image.open(base/'masks'/(stem+'.png')) as ann:
                if im.format!='TIFF' or ann.format!='PNG' or im.mode!='RGB' or ann.mode!='RGB': raise ValueError('Original format mismatch')
                rgb=np.array(im); target=semantic_binary(rgb,np.array(ann),s['coco_categories'])
                if int(target.sum())!=s['source_positive_pixels']: raise ValueError('Source label count differs')
            validity(rgb,p['review'])
            with rasterio.open(base/'images'/(stem+'.tif')) as src:
                if str(src.crs)!=s['source_crs'] or list(src.transform)[:6]!=s['source_transform'] or list(src.shape)!=s['input_shape'] or list(src.bounds)!=s['source_bounds']:
                    raise ValueError('Original source grid differs')
                if src.nodata is not None or not np.all(src.dataset_mask()==255): raise ValueError('Original validity differs')
            check_products(root,s)
            s.update(status=ACCEPTED,official_split='train',acceptance_evidence=dict(ledger_pointer=stage+'/result/samples/image_id='+str(rid),
                run_manifest_sha256=expected,review_receipt_sha256=run['evidence_sha256'],review_basis=p['review']['basis'],
                review_preview_sha256=p['review']['preview_sha256'],source_checks='original SHA/grid/PNG label, STAC rights, source-bound visual receipt, product SHA/grid/threshold/validity PASS',
                separation=leak))
            samples.append(s); source_roots[rid]=root
    manifest=dict(schema='ouem-m1-7-g0-v1',source_generation=GENERATION,acceptance=ACCEPTED,
                  ledger_sha256=sha(ledger_path),scope='tiny functional dataset only; no accuracy/commercial clearance claim',
                  samples=sorted(samples,key=lambda s:s['image_id']))
    manifest['dataset_id']=identify(manifest);check_manifest(manifest)
    fresh(output,[zenodo,d2_root,d3_root,d2_evidence.parent,d3_evidence.parent])
    for s in manifest['samples']:
        folder=output/s['split']/str(s['image_id']);folder.mkdir(parents=True)
        for name,digest in s['output_hashes'].items():
            source=source_roots[s['image_id']]/s['split']/str(s['image_id'])/name
            require_hash(source,digest);shutil.copyfile(source,folder/name)
            require_hash(folder/name,digest);require_hash(source,digest)
    destination=output/'manifest.json';destination.write_bytes(encoded(manifest))
    result=verify(output,destination,sha(destination))
    (output/'manifest.sha256').write_text(result['manifest_sha256']+'  manifest.json\n',encoding='ascii')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    v=sub.add_parser('verify');v.add_argument('--root',type=Path,required=True);v.add_argument('--manifest',type=Path,required=True)
    v.add_argument('--expected-sha256',required=True,help='Trusted Git receipt; do not trust a transferred sidecar alone')
    f=sub.add_parser('freeze')
    for name in ['zenodo','d2-root','d3-root','d2-evidence','d3-evidence','output']:f.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('pack');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    u=sub.add_parser('unpack')
    u.add_argument('--archive',type=Path,required=True);u.add_argument('--expected-zip-sha256',required=True)
    u.add_argument('--output',type=Path,required=True);u.add_argument('--result-json',type=Path,required=True)
    args=vars(parser.parse_args());stage=args.pop('stage')
    if stage=='verify': result=verify(args['root'],args['manifest'],args['expected_sha256'])
    elif stage=='freeze': result=freeze(**args)
    elif stage=='pack': result=pack(**args)
    else: result=unpack_report(**args)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
