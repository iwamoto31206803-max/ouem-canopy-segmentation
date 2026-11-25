"""A3-2 admission and approval checks for the bounded research pilot."""
from collections import Counter
import json
from pathlib import Path
import sys
from d6a3_produce import AUTH, RECEIPT, identifier, need, pinned, plain, source_paths
import d6a_research_qa as qa
from m1_5b_pilot import separation
from r1_dataset import safe_path


PLAN = 'd6-real-pilot-plan-v1'
APPROVAL = 'd6-real-pilot-approval-v1'
GT54 = ['GT54_EXTERNAL_EVALUATION_DEFERRED', 'GT54_GEOGRAPHIC_INDEPENDENCE_UNVERIFIED']
FREEZE_SHA = 'd7a72bdf1c792678cee0030c72dded18d3cdd6639a800c12ebafc63ba3fa1978'
SUBINDEX_SHA = '079d37349734c746038bab90144babf50ae75d20aacfbf8cda61b5aa1bd84dae'
CROP_PLAN_SHA = 'b6711c33eafed51dd8d42040abf13e3a52601d5be54b08479217ac7e043c8692'
STORAGE_POLICY = 'BOUNDED_WRITES_AND_OBSERVED_ALLOCATION'
LOCAL_SHA = 'SOURCE_AUTHENTICITY_NOT_EXTERNALLY_VERIFIED'
MAXIMUMS = dict(commit_bytes=2560*1024**2, rss_bytes=2*1024**3, seconds=120,
                attempt_seconds=600, temporary_bytes=4*1024**2,
                output_bytes=16*1024**2, physical_bytes=64*1024**2)
MINIMUMS = dict(host_available_bytes=6*1024**3, free_bytes=256*1024**2)


def approval(plan_path, plan_sha, approval_path, approval_sha):
    plan_path, approval_path = map(plain, (plan_path, approval_path))
    need(plan_path.stat().st_size<=65536 and approval_path.stat().st_size<=4096,
         'REAL_CONTROL_FILE_SIZE')
    plan = pinned(plan_path, plan_sha)
    token = pinned(approval_path, approval_sha)
    need(set(token) == {'schema','approved','plan_sha256','purpose'} and token['approved'] is True
         and token == dict(schema=APPROVAL,approved=True,plan_sha256=plan_sha,
                           purpose='ONE_BOUNDED_REAL_OUTPUT_PILOT'), 'REAL_APPROVAL_BINDING')
    need(plan['schema']==PLAN and plan['source_generation']==qa.GENERATION
         and plan['storage_policy']==STORAGE_POLICY and plan['gt54']==GT54 and plan['windows_per_source']=={'train':1,'val':1},
         'REAL_PLAN_SCOPE')
    limits=plan['limits']
    need(set(limits)==set(MAXIMUMS)|set(MINIMUMS)
         and all(type(limits[k]) is int and 0<limits[k]<=v for k,v in MAXIMUMS.items())
         and all(type(limits[k]) is int and limits[k]>=v for k,v in MINIMUMS.items())
         and limits['temporary_bytes']<=limits['output_bytes']<=limits['physical_bytes']
         and limits['rss_bytes']<=limits['commit_bytes'] and limits['seconds']<=limits['attempt_seconds'],
         'REAL_RESOURCE_LIMITS')
    sources=plan['sources']
    roles=Counter(s['split'] for s in sources)
    need(set(roles)=={'train','val'} and all(1<=n<=2 for n in roles.values())
         and all(type(s['image_id']) is int for s in sources)
         and len({s['image_id'] for s in sources})==len(sources), 'REAL_COHORT_LIMIT')
    need(plan['allow_authenticity'] and set(plan['allow_authenticity'])<=AUTH,'REAL_AUTHENTICITY_POLICY')
    need(sum(s.get('source_hash_verification') == LOCAL_SHA for s in sources) <= 3,
         'REAL_LOCAL_ONLY_LIMIT')
    return plan


def protected_lineage(row, group, protected):
    """Use known catalog envelopes; preserve exact same-role historical Train reuse."""
    alias=qa.identity(row)
    reused=False
    for previous in protected:
        role=previous['split']
        box=previous['envelope']
        need(role in ('train','val') and qa.digest(previous['alias'])
             and -180<=box[0]<box[2]<=180 and -90<=box[1]<box[3]<=90,
             'OAM_PROTECTED_EVIDENCE')
        match=(previous['oam_id'].casefold()==row['oam_id'].casefold() or previous['alias']==alias)
        if match:
            need(role==row['split']=='train' and previous['alias']==alias
                 and previous['oam_id']==row['oam_id'],'REAL_PROTECTED_IDENTITY')
            reused=True
        elif role=='val' or row['split']=='val':
            center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
            separation(dict(row,wgs84_bounds=group['wgs84_bounds'],center=group['center']),
                       [dict(image_id=-1,oam_id=previous['oam_id'],official_split='test',
                             wgs84_bounds=box,center=center)])
    return reused


def verify_evidence(plan, plan_sha, base, root, evidence):
    """Validate metadata and hash input bytes without decoding rasters."""
    def ref(link):
        return pinned(safe_path(base,link['path']),link['sha256'])
    need(plan['protected']['sha256']==SUBINDEX_SHA
         and plan['historical_freeze']['sha256']==FREEZE_SHA, 'HISTORICAL_ANCHOR')
    protected=ref(plan['protected'])
    prior=protected['sources']
    need(protected['scope']=='v5 and nested historical Val18 only'
         and Counter(p['split'] for p in prior)=={'train':70,'val':18}
         and len({p['image_id'] for p in prior})==88
         and len({p['alias'] for p in prior})==88
         and len({p['oam_id'].casefold() for p in prior})==88,'OAM_PROTECTED_EVIDENCE')
    freeze=ref(plan['historical_freeze'])
    need(plan['historical_freeze']['sha256']==protected['provenance']['freeze_sha256']
         and freeze['identity']['crop_plan_sha256']==protected['provenance']['crop_plan_sha256']==CROP_PLAN_SHA,
         'HISTORICAL_BINDING')
    paths={k:safe_path(root,v) for k,v in qa.FILES.items()}
    need(set(plan['catalog_sha256'])==paths.keys(),'REAL_CATALOG_BINDING')
    for k,path in paths.items():
        qa.require_hash(path,plan['catalog_sha256'][k])
    need(protected['provenance']['metadata_sha256']=={
         f'{k}_meta.json':plan['catalog_sha256'][k] for k in ('train','test')},'HISTORICAL_CATALOG_BINDING')
    rows,groups,objects=qa.load_catalog(paths)
    for previous in prior:
        row=rows[previous['image_id']]
        need(row['official_split']=='train' and row['split']==previous['split']
             and row['oam_id']==previous['oam_id'] and qa.identity(row)==previous['alias']
             and groups[row['oam_id']]['wgs84_bounds']==previous['envelope'], 'OAM_PROTECTED_EVIDENCE')
    refs={}
    for role in ('train','val'):
        pair=plan['a1'][role]
        contract,report=ref(pair['contract']),ref(pair['report'])
        need(contract['schema']==report['policy']==qa.POLICY
             and contract['source_generation']==report['source_generation']==qa.GENERATION
             and contract['catalog_sha256']==report['catalog_sha256']==plan['catalog_sha256']
             and type(contract['seed']) is int and 0<=contract['seed']<2**32
             and type(report['seed']) is int and contract['seed']==report['seed']
             and report['contract_sha256']==pair['contract']['sha256'],'REAL_A1_BINDING')
        candidates={v['image_id']:v for v in contract['candidates']}
        outcomes={v['image_id']:v for v in report['results']}
        need(len(candidates)==len(contract['candidates']) and len(outcomes)==len(report['results'])
             and set(candidates)==set(outcomes)
             and all(type(v['image_id']) is int and type(v['validation_fold']) is int
                     for v in [*contract['candidates'], *report['results']]), 'REAL_A1_COHORT')
        eligible={'train':[], 'val':[]}
        for rid, a in sorted(outcomes.items()):
            row, c = rows[rid], candidates[rid]
            need(all(row[k]==a[k]==c[k] for k in ('oam_id','validation_fold'))
                 and all(row[k]==a[k] for k in ('split','official_split'))
                 and a['final_decision']==qa.final_decision(a['source_decision'],a['window_decision']),
                 'REAL_A1_RESULT_BINDING')
            if a['final_decision']['status']==qa.ELIGIBLE:
                need(row['official_split']=='train','REAL_A1_OFFICIAL_TRAIN')
                eligible[row['split']].append(rid)
        need(report['eligible_ids']==eligible
             and report['counts']==dict(Counter(a['final_decision']['status'] for a in outcomes.values()))
             and type(report['catalog_rows']) is int and report['catalog_rows']==len(rows)
             and report['crop_policy']==qa.CROP_POLICY and report['labels']==qa.LABEL_POLICY,
             'REAL_A1_ALLOWLIST')
        refs[role]=(candidates,outcomes,eligible[role])
    annotations=qa.coco_annotation_summary(paths['coco'], [s['image_id'] for s in plan['sources']])
    sources=[]
    seen=set()
    for s in plan['sources']:
        row=rows[s['image_id']]
        role,oid=s['split'],s['oam_id']
        alias=qa.identity(row)
        need(row['official_split']=='train' and role==row['split'] and oid==row['oam_id']
             and type(s['validation_fold']) is int and s['validation_fold']==row['validation_fold']
             and s['source_object_sha256']==alias and alias not in seen and oid.casefold() not in seen
             and type(row['width']) is int and type(row['height']) is int
             and 0<row['width']*row['height']<=qa.MAX_PIXELS,'REAL_SOURCE_IDENTITY')
        seen.update((alias,oid.casefold()))
        candidates,outcomes,eligible=refs[role]
        a,c=outcomes[s['image_id']],candidates[s['image_id']]
        reuse=protected_lineage(row,groups[oid],prior)
        need(s['historical_train_reuse'] is reuse and a['final_decision']['status']==qa.ELIGIBLE
             and s['image_id'] in eligible and a['split']==role
             and a['official_split']=='train' and a['oam_id']==c['oam_id']==oid
             and a['validation_fold']==c['validation_fold']==row['validation_fold']
             and type(a['selected_black_support_cells']) is int and a['selected_black_support_cells']==0
             and a['coco_categories']==annotations.get(s['image_id'],{}).get('categories',[])
             and all(s[k]==a[k] for k in ('output_shape','crop_transform'))
             and all(s[k]==a[k] for k in ('crop_xywh','source_hashes','source_hash_verification',
                                           'source_object_sha256','stac_sha256')), 'REAL_A1_SOURCE_BINDING')
        crop=s['crop_xywh']
        shape=s['output_shape']
        need(len(shape)==2 and all(type(v) is int and v>0 for v in shape)
             and 224<=shape[0]<=row['height'] and 224<=shape[1]<=row['width']
             and len(s['crop_transform'])==6 and all(type(v) in (int,float) for v in s['crop_transform']),
             'REAL_OUTPUT_GRID')
        need(len(crop)==4 and all(type(v) is int for v in crop) and min(crop[:2])>=0
             and crop[2:]==[224,224] and crop[0]+224<=a['output_shape'][1]
             and crop[1]+224<=a['output_shape'][0],'REAL_CROP')
        status=s['source_hash_verification']; hashes=s['source_hashes']
        need(status in plan['allow_authenticity'] and set(hashes)=={'rgb','annotation'}
             and all(qa.digest(v) for v in hashes.values())
             and (c.get('source_hashes')==a.get('expected_source_hashes')==hashes
                  if status=='MATCHED_EXTERNAL_EXPECTED_SHA256'
                  else 'source_hashes' not in c and 'expected_source_hashes' not in a),'REAL_AUTHENTICITY_BINDING')
        safe_path(evidence,oid+'-stac.json')
        try:
            verified=qa.source_preflight(row,c,groups,objects,set(),evidence)
        except qa.Decision as exc:
            raise ValueError(exc.result['reason']) from exc
        need(all(verified[k]==a[k] for k in verified),'REAL_STAC_BINDING')
        need({k:qa.sha(p) for k,p in source_paths(root,row).items()}==hashes,'INPUT_PIN_MISMATCH')
        key=identifier([qa.GENERATION,s['image_id'],oid,alias])
        sources.append(dict(schema=RECEIPT,plan_sha256=plan_sha,source_key=key,
                            patch_id=identifier([plan_sha,key,crop]),image_id=s['image_id'],
                            oam_id=oid,source_object_sha256=alias,split=role,
                            validation_fold=row['validation_fold'],crop_xywh=crop,input_sha256=hashes,
                            stac_sha256=s['stac_sha256'],source_hash_verification=status,
                            historical_train_reuse=reuse,final_decision=qa.ELIGIBLE,
                            a1_output_shape=a['output_shape'],a1_crop_transform=a['crop_transform'],
                            categories=a['coco_categories']))
    return plan['limits'],sources,rows

def execution_scope(plan, plan_path, token, root, evidence, output):
    from d6a3_produce import output_scope
    from d6a3_guard import require_local_fixed

    locations=dict(cache=root, evidence=evidence, output=output)
    need(set(plan['locations'])==locations.keys()
         and all(Path(plan['locations'][k]).absolute()==v for k,v in locations.items()),
         'REAL_LOCATION_BINDING')
    protected=tuple(Path(p) for p in plan['protected_roots'])
    require_local_fixed(plan_path,token,root,evidence,output,*protected)
    protected=tuple(map(plain,protected))
    need(protected and all(p.is_dir() for p in protected), 'REAL_PROTECTED_ROOTS')
    output_scope(plan_path,root,evidence,output,(*protected,token))
    need(output.parent.is_dir() and plain(output.parent).is_dir(), 'REAL_OUTPUT_PARENT')


def execute(plan_path, plan_sha, token, token_sha, root, evidence, output, *, resume=False):
    from d6a3_guard import require_local_fixed, watch

    path_args=(plan_path,token,root,evidence,output)
    require_local_fixed(*path_args)
    plan_path,token,root,evidence,output=map(plain,path_args)
    plan=approval(plan_path,plan_sha,token,token_sha)
    execution_scope(plan,plan_path,token,root,evidence,output)
    return watch((plan_path,plan_sha,token,token_sha,root,evidence,output,int(resume)),
                 plan['limits'],attempt=True)


def run_guarded(plan_path, plan_sha, token, token_sha, root, evidence, output, resume):
    from d6a3_produce import offline_worker, publish
    from r1d_bounded_prepare import WindowsJob

    try:
        offline_worker()
        plan_path,token,root,evidence,output=map(lambda p: plain(Path(p)),
                                                  (plan_path,token,root,evidence,output))
        plan=approval(plan_path,plan_sha,token,token_sha)
        WindowsJob.require_current(plan['limits']['commit_bytes'])
        execution_scope(plan,plan_path,token,root,evidence,output)
        inputs=verify_evidence(plan,plan_sha,plan_path.parent,root,evidence)
        result=publish(plan_path,plan_sha,root,output,inputs,resume=resume=='1',real=True)
        print(json.dumps(result))
    except qa.Decision as exc:
        print(f"D6 pilot stopped: {exc.result['reason']}", file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        # Exception messages may contain private paths or source identifiers.
        reason = str(exc)
        if reason.startswith('RESOURCE_STOP:') or (reason.isascii() and reason.replace('_','').isalnum()):
            print(f'D6 pilot stopped: {reason}', file=sys.stderr)
        else:
            print(f'D6 pilot stopped: {type(exc).__name__}; inspect private inputs or output', file=sys.stderr)
        raise SystemExit(1) from None
