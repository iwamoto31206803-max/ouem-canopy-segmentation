"""R1b-1 metadata-only, offline candidate and resource planning; never admits data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from m1_5a_inventory import role
from m1_5b_pilot import separation, rights
from m1_5d_zenodo_prepare import local_index
from m1_7_dataset import GENERATION, MANIFEST_SHA256, encoded, sha, require_hash

MIB = 1024**2
KNOWN_HOLD = {1,2,9,576,1401,4235}


def identity(row):
    """Bucket-independent original metadata object key; no private URL in report."""
    value = row.get('meta')
    if not isinstance(value, str) or urlparse(value).scheme not in ('http','https'):
        return None
    path = urlparse(value).path
    return hashlib.sha256(path.encode('utf-8')).hexdigest() if path and path != '/' else None


def held_ids(ledger):
    result = set(KNOWN_HOLD)
    def walk(value):
        if isinstance(value, dict):
            if type(value.get('image_id')) is int and str(value.get('status','')).startswith('HOLD'):
                result.add(value['image_id'])
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(ledger)
    return result


def cached_rights(row, evidence_dirs):
    snapshots = []
    for folder in evidence_dirs:
        p = folder/(row['oam_id']+'-stac.json')
        if p.is_file(): snapshots.append((sha(p),folder))
    if not snapshots: return dict(status='HOLD',reason='STAC_EVIDENCE_MISSING')
    if len({h for h,_ in snapshots}) != 1:
        return dict(status='HOLD',reason='CONFLICTING_STAC_SNAPSHOTS')
    digest, folder = snapshots[0]
    try:
        checked = dict(row,source_evidence=dict(snapshot_sha256=digest,url='https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+row['oam_id']))
        _, receipt = rights(checked,folder)
        return dict(status='CACHED_DECLARATION_CHECKED',stac_sha256=digest,
                    image_license=receipt['image_license'],annotation_license=receipt['annotation_license'],
                    annotation_terms=receipt['annotation_terms'],
                    attribution='Retained in external hashed STAC; R1b human attribution review pending')
    except (ValueError,KeyError,OSError,TypeError) as exc:
        # Do not expose arbitrary external field contents or contact data.
        return dict(status='HOLD',reason='STAC_LICENSE_ID_ATTRIBUTION_OR_OBJECT_KEY_INVALID',exception_type=type(exc).__name__)


def select(rows, protected, holds, seed=0, pool_size=150, batch_size=25, evidence_dirs=()):
    if type(pool_size) is not int or type(batch_size) is not int or not 1 <= pool_size <= 200 or not 20 <= batch_size <= 30 or batch_size > pool_size:
        raise ValueError('Pool <=200 and pilot 20..30 within pool required')
    if type(seed) is not int: raise ValueError('Integer seed required')
    if len({r['image_id'] for r in rows}) != len(rows): raise ValueError('Duplicate image ID')
    indexed = {r['image_id']:r for r in rows}
    if any(i not in indexed or role(indexed[i]) != split for i,split in protected.items()):
        raise ValueError('Legacy protected roles missing/changed')
    blocked = {r['oam_id'] for r in rows if r['image_id'] in set(protected)|holds}
    blocked_objects = {identity(r) for r in rows if r['oam_id'] in blocked} - {None}
    by_oam = defaultdict(list)
    for row in rows: by_oam[row['oam_id']].append(row)
    catalog = []; invalid_groups = set()
    for group, members in sorted(by_oam.items()):
        if len({(r['official_split'],r.get('validation_fold'),role(r)) for r in members}) != 1:
            invalid_groups.add(group)
        located = [r['wgs84_bounds'] for r in members if r.get('wgs84_bounds')]
        item = dict(members[0])
        if len(located) == len(members):
            b = [min(x[0] for x in located),min(x[1] for x in located),max(x[2] for x in located),max(x[3] for x in located)]
            item.update(wgs84_bounds=b,center=[(b[0]+b[2])/2,(b[1]+b[3])/2])
        else: item.pop('wgs84_bounds',None)
        catalog.append(item)
    group_info = {r['oam_id']:r for r in catalog}
    object_roles = defaultdict(set)
    for row in rows:
        if identity(row): object_roles[identity(row)].add(role(row))
    reasons = {}; eligible = []
    for r in sorted(rows,key=lambda r:r['image_id']):
        rid = r['image_id']; reason = None
        if r['official_split'] != 'train': reason='EXCLUDED_OFFICIAL_TEST_HOLDOUT'
        elif rid in protected: reason='SEPARATE_LEGACY_SMOKE_NOT_R1_ACCEPTED'
        elif r['oam_id'] in blocked or identity(r) in blocked_objects: reason='EXCLUDED_PROTECTED_OR_KNOWN_HOLD_SOURCE'
        elif r.get('license') != 'CC-BY 4.0': reason='HOLD_NONEXPLICIT_CC_BY_4_0'
        elif type(r.get('validation_fold')) is not int or r['validation_fold'] not in range(5): reason='HOLD_FOLD'
        elif not identity(r): reason='HOLD_MISSING_ORIGINAL_SOURCE_IDENTITY'
        elif r['oam_id'] in invalid_groups or len(object_roles[identity(r)]) != 1: reason='HOLD_SOURCE_ROLE_FOLD_CONFLICT'
        elif not group_info[r['oam_id']].get('wgs84_bounds'): reason='HOLD_MISSING_GEOGRAPHY'
        elif any(r.get(k) is None for k in ('width','height','crs','biome')) or any(type(r.get(k)) is not int or r[k]<=0 for k in ('width','height')): reason='HOLD_REQUIRED_METADATA'
        if reason: reasons[rid]=reason
        else: eligible.append(r)
    # Rank one representative per original object, independent of pixels/predictions.
    def rank(r): return hashlib.sha256(f"{seed}:{r['oam_id']}:{r['image_id']}".encode()).hexdigest()
    representatives = {}; used_groups=set()
    for r in sorted(eligible,key=lambda r:(rank(r),r['image_id'])):
        key=identity(r)
        if key in representatives or r['oam_id'] in used_groups:
            reasons[r['image_id']]='EXCLUDED_DUPLICATE_SOURCE_REPRESENTATIVE';continue
        representatives[key]=r;used_groups.add(r['oam_id'])
    remaining=[]
    for r in representatives.values():
        try:
            # Reuse pilot separation against ALL opposite source groups, including test metadata.
            candidate=dict(r,wgs84_bounds=group_info[r['oam_id']]['wgs84_bounds'],center=group_info[r['oam_id']]['center'])
            guard=separation(candidate,catalog)
            remaining.append((r,guard))
        except (ValueError,KeyError,TypeError) as exc:
            reasons[r['image_id']]='HOLD_SPATIAL_OR_SOURCE_RISK:'+str(exc)
    cells=Counter(); biomes=Counter(); chosen=[]
    for slot in range(pool_size):
        desired='val' if slot%5==4 else 'train'
        available=[pair for pair in remaining if role(pair[0])==desired]
        if not available: continue
        def score(pair):
            r=pair[0];lon,lat=r['center'];cell=(math.floor(lon/10),math.floor(lat/10))
            return cells[cell],biomes[str(r['biome'])],rank(r),r['image_id']
        r,guard=min(available,key=score);remaining.remove((r,guard))
        lon,lat=r['center'];cell=(math.floor(lon/10),math.floor(lat/10))
        cells[cell]+=1;biomes[str(r['biome'])]+=1
        evidence=cached_rights(r,evidence_dirs)
        chosen.append(dict(image_id=r['image_id'],oam_id=r['oam_id'],split=role(r),validation_fold=r['validation_fold'],row_license=r['license'],official_split=r['official_split'],
            original_source_identity_sha256=identity(r),
            representative_metadata_center=r['center'],representative_metadata_bounds_wgs84=r['wgs84_bounds'],
            group_metadata_center=group_info[r['oam_id']]['center'],
            group_metadata_bounds_wgs84=group_info[r['oam_id']]['wgs84_bounds'],
            group_metadata_member_count=len(by_oam[r['oam_id']]),
            geographic_cell=list(cell),biome=r['biome'],shape=[r['height'],r['width']],rights=evidence,
            metadata_separation=guard,canopy_fraction='NOT_MEASURED',image_conditions='NOT_VERIFIED',
            status='HOLD',pending=['SOURCE_BYTE_AND_ARCHIVE_AUTHENTICATION','RIGHTS_AND_ATTRIBUTION_REVIEW',
                'ORIGINAL_SOURCE_EXTENT_AND_EXTERNAL_CATALOG_REVIEW','CATEGORY_VALIDITY_ALIGNMENT_AND_VISUAL_QA'],
            selection_reason='Prefer least-used 10-degree cell, then biome, then seeded SHA rank; 4 Train : 1 Val slots'))
        reasons[r['image_id']]='SELECTED_PROVISIONAL_HOLD'
    for r,_ in remaining: reasons[r['image_id']]='RESERVE_NOT_SELECTED_POOL_LIMIT'
    return dict(seed=seed,requested_pool=pool_size,requested_batch=batch_size,pool=chosen,
                completeness=completeness(chosen,pool_size,batch_size),
                initial_batch_ids=[c['image_id'] for c in chosen[:batch_size]],
                counts=dict(Counter(c['split'] for c in chosen)),
                reasons_by_image={str(k):v for k,v in sorted(reasons.items())},
                reason_counts=dict(Counter(reasons.values())),geographic_cells=len(cells),biomes=len(biomes),
                external_independence='NOT_VERIFIED: TCD group envelopes checked; full original/STAC coverage not established')


def completeness(chosen, pool_size, batch_size):
    """Validate the actual prefix, not merely the nominal selection-slot count."""
    result={};reasons=[]
    for name,n,actual in [('pool',pool_size,chosen),('initial_batch',batch_size,chosen[:batch_size])]:
        expected={'train':n-n//5,'val':n//5}
        counts={split:sum(c['split']==split for c in actual) for split in ('train','val')}
        ok=len(actual)==n and counts==expected
        result[name]=dict(status='PASS' if ok else 'HOLD',expected_count=n,actual_count=len(actual),
                          expected_split_counts=expected,actual_split_counts=counts)
        if not ok:reasons.append(name.upper()+'_COUNT_OR_SPLIT_SHORTFALL')
    return dict(status='HOLD' if reasons else 'PASS',reasons=reasons,**result)


def measurements(pair_dirs, preview_dirs, stac_dirs):
    pairs=[]; previews=[]; documents=[]; stac=[]; references=[]
    for directory in pair_dirs:
        for path in sorted(directory.rglob('rgb.tif')):
            files=[path.with_name(n) for n in ('rgb.tif','canopy_fraction.tif','canopy_mask.tif','valid_fraction.tif')]
            if all(p.is_file() for p in files): pairs.append(sum(p.stat().st_size for p in files))
        for p in sorted(directory.rglob('*.json')):
            documents.append(p.stat().st_size)
            references.append(dict(kind='prior_pair_manifest',name=directory.name+'/'+p.name,sha256=sha(p)))
    for directory in preview_dirs:
        previews.extend(p.stat().st_size for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.png'))
    for directory in stac_dirs:
        stac.extend(p.stat().st_size for p in directory.glob('*-stac.json'))
    def stats(values):
        return dict(count=len(values),total_bytes=sum(values),max_bytes=max(values) if values else None)
    return dict(pair_products=stats(pairs),previews=stats(previews),documents=stats(documents),stac=stats(stac),references=references)


def resources(root, batch, measured, plan_bytes=0):
    sizes=[]; missing=[]
    for c in batch:
        stem=c['oam_id']+'_'+str(c['image_id']);base=root/'extracted/dataset/holdout'
        files=[base/'images'/(stem+'.tif'),base/'masks'/(stem+'.png')]
        if all(p.is_file() for p in files): sizes.append(dict(image_id=c['image_id'],rgb_bytes=files[0].stat().st_size,annotation_bytes=files[1].stat().st_size))
        else: missing.append(c['image_id'])
    valid=all(measured[k]['count'] for k in ('pair_products','previews','documents','stac')) and not missing
    estimate=None
    if valid:
        n=len(batch);pair=measured['pair_products']['max_bytes'];preview=measured['previews']['max_bytes'];doc=measured['documents']['max_bytes']
        # Two distinct deterministic output runs, source/preview temporary files and 2x headroom.
        report_allowance=2*plan_bytes+2*n*doc
        subtotal=2*n*pair+n*preview+report_allowance+n*65536+max((s['rgb_bytes']+s['annotation_bytes'] for s in sizes),default=0)+pair
        estimate=dict(two_output_runs_bytes=2*n*pair,qa_preview_bytes=n*preview,report_allowance_bytes=report_allowance,
                      stac_snapshot_allowance_bytes=n*65536,
                      temporary_disk_allowance_bytes=max((s['rgb_bytes']+s['annotation_bytes'] for s in sizes),default=0)+pair,
                      modeled_incremental_peak_bytes=subtotal,safety_factor=2,
                      proposed_incremental_storage_limit_bytes=math.ceil(2*subtotal/MIB)*MIB)
    return dict(mode='READ_ONLY_EXISTING_ZENODO_CACHE',source_file_sizes=sizes,missing_local_ids=missing,
                source_rgb_bytes=sum(s['rgb_bytes'] for s in sizes),source_annotation_bytes=sum(s['annotation_bytes'] for s in sizes),
                image_annotation_network_payload_bytes=0 if not missing else 'NOT_ESTIMATED_STOP',
                http_range_cache_bytes=0,parquet_acquisition='DISABLED; historical HF row-group cost not extrapolated to Zenodo',
                proposed_metadata_transfer_limit_bytes=2*MIB,max_stac_requests=len(batch),max_single_stac_body_bytes=65536,
                incremental_disk_estimate=estimate,estimate_status='PROPOSED_NOT_PEAK_MEASURED' if valid else 'HOLD_MISSING_MEASUREMENTS',
                no_automatic_archive_download=True,bootstrap_archive_sizes={p.name:p.stat().st_size for p in (root/'archives').glob('*.tar.gz')},
                stop_policy='Missing local data or bytes exceeding ceilings: STOP, no HF fallback, no cap reset, no candidate substitution')


def verify_metadata(root, ledger):
    prior=ledger['m1_5d2']['result']
    if prior['source_generation'] != GENERATION:raise ValueError('Historical metadata generation mismatch')
    for name in ('train_meta.json','test_meta.json'):
        require_hash(root/'archives'/name,prior['metadata_hashes'][name])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zenodo-root',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--g0-manifest',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,action='append',default=[])
    p.add_argument('--pairs-dir',type=Path,action='append',default=[])
    p.add_argument('--preview-dir',type=Path,action='append',default=[])
    p.add_argument('--pool-size',type=int,default=150)
    p.add_argument('--batch-size',type=int,default=25)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--supersedes-plan-id',help='Optional reviewed predecessor plan; does not grant acceptance')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('Fresh plan report required')
    destination=a.output.resolve()
    for directory in [a.zenodo_root,*a.evidence_dir,*a.pairs_dir,*a.preview_dir]:
        if destination == directory.resolve() or directory.resolve() in destination.parents:
            raise ValueError('Plan output overlaps protected input directory')
    require_hash(a.g0_manifest,MANIFEST_SHA256)
    g0=json.loads(a.g0_manifest.read_text());ledger=json.loads(a.ledger.read_text())
    verify_metadata(a.zenodo_root,ledger)
    rows=local_index(a.zenodo_root)
    plan=select(rows,{s['image_id']:s['split'] for s in g0['samples']},held_ids(ledger),a.seed,a.pool_size,a.batch_size,a.evidence_dir)
    measured=measurements(a.pairs_dir,a.preview_dir,a.evidence_dir)
    initial=set(plan['initial_batch_ids'])
    report=dict(schema='ouem-r1b1-plan-v1',source_generation=GENERATION,zenodo_record='https://zenodo.org/records/11617167',
        inputs=dict(metadata_sha256={name:sha(a.zenodo_root/'archives'/name) for name in ('train_meta.json','test_meta.json')},
                    legacy_g0_manifest_sha256=MANIFEST_SHA256,legacy_qa_ledger_sha256=sha(a.ledger)),
        catalog=dict(images=len(rows),source_groups=len({r['oam_id'] for r in rows}),official_train=sum(r['official_split']=='train' for r in rows)),
        plan=plan,empirical_measurements=measured,
        resources=resources(a.zenodo_root,[c for c in plan['pool'] if c['image_id'] in initial],measured,len(encoded(plan))),
        legacy_smoke=dict(ids=sorted(s['image_id'] for s in g0['samples']),r1_acceptance='UNCHANGED_HOLD'),
        known_hold_ids=sorted(held_ids(ledger)),external_network=dict(http_requests=0,payload_bytes=0,scope='This metadata planner only; Git synchronization is not dataset HTTP traffic'),
        gate='HOLD_PENDING_RIGHTS_EXTERNAL_INDEPENDENCE_QA_AND_BUDGET_REVIEW',
        pixels_opened=0,training_acceptances=0)
    history=ledger['m1_5b3']['acquisition']
    report['historical_http_range_example']=dict(payload_bytes=history['network_payload_bytes'],
        compressed_column_bytes=history['column_bytes'],largest_compressed_group_bytes=history['largest_compressed_group_bytes'],
        saved_bytes_at_observation=history['all_saved_bytes_at_observation'],peak_disk_usage=history['peak_disk_usage'],
        use_for_zenodo_per_image_extrapolation=False)
    report['r1b2_execution_ready']=False
    report['r1b2_blockers']=['REVIEW_AND_APPROVE_BUDGET','BOUNDED_PERSISTENT_LEDGER_AND_NEW_COHORT_SCREEN_WRAPPER_NOT_IMPLEMENTED',
        'PER_IMAGE_RIGHTS_AND_EXTERNAL_SOURCE_REVIEW','ALL_INITIAL_CANDIDATES_REQUIRE_VISUAL_QA']
    if plan['completeness']['status']!='PASS':
        report['gate']='HOLD_INCOMPLETE_COHORT'
        report['r1b2_blockers'].append('CANDIDATE_COUNT_OR_SPLIT_SHORTFALL')
    report['resource_execution_gate']=dict(status='HOLD',actual_batch_peak_measured=False,
        requirements=['R1B2_MEASURE_ACTUAL_RETAINED_AND_TEMPORARY_BYTES',
                      'RESERVE_BEFORE_HTTP_OR_WRITE_AND_STOP_BEFORE_CUMULATIVE_LIMIT',
                      'NO_BUDGET_RESET_OR_AUTOMATIC_INCREASE'])
    if a.supersedes_plan_id:
        report['supersedes_plan_id']=a.supersedes_plan_id
        report['revision_reason']='PR43 review: actual cohort/split validation, explicit representative/group extents, measured-resource execution gate'
    report['plan_id']='ouem-r1b1-'+hashlib.sha256(encoded(report)).hexdigest()
    with a.output.open('xb') as stream:stream.write(encoded(report))
    print(json.dumps(dict(plan_id=report['plan_id'],pool=len(plan['pool']),initial_batch=len(plan['initial_batch_ids']),gate=report['gate'])))
    return 2 if plan['completeness']['status']=='HOLD' else 0


if __name__=='__main__':raise SystemExit(main())
