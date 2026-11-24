"""A3-2 admission and approval checks. Real output remains blocked."""
from collections import Counter
from d6a3_produce import AUTH, need, pinned, plain
import d6a_research_qa as qa
from m1_5b_pilot import separation
from r1_dataset import safe_path


PLAN = 'd6-real-pilot-plan-v1'
APPROVAL = 'd6-real-pilot-approval-v1'
GT54 = ['GT54_EXTERNAL_EVALUATION_DEFERRED', 'GT54_GEOGRAPHIC_INDEPENDENCE_UNVERIFIED']
MAXIMUMS = dict(commit_bytes=2560*1024**2, rss_bytes=2*1024**3, seconds=120,
                attempt_seconds=600, temporary_bytes=4*1024**2,
                output_bytes=16*1024**2, physical_bytes=64*1024**2)
MINIMUMS = dict(host_available_bytes=6*1024**3, free_bytes=256*1024**2)


def approval(plan_path, plan_sha, approval_path, approval_sha):
    plan = pinned(plain(plan_path), plan_sha)
    token = pinned(plain(approval_path), approval_sha)
    need(set(token) == {'schema','approved','plan_sha256','purpose'} and token['approved'] is True
         and token == dict(schema=APPROVAL,approved=True,plan_sha256=plan_sha,
                           purpose='ONE_BOUNDED_REAL_OUTPUT_PILOT'), 'REAL_APPROVAL_BINDING')
    need(plan['schema']==PLAN and plan['source_generation']==qa.GENERATION
         and plan['gt54']==GT54 and plan['windows_per_source']=={'train':1,'val':1},
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


def verify_evidence(plan, base, root, evidence):
    """Metadata-only future preflight; no raster opens or writes."""
    def ref(link):
        return pinned(safe_path(base,link['path']),link['sha256'])
    protected=ref(plan['protected'])
    prior=protected['sources']
    need(protected['scope']=='v5 and nested historical Val18 only'
         and Counter(p['split'] for p in prior)=={'train':70,'val':18}
         and len({p['alias'] for p in prior})==88
         and len({p['oam_id'].casefold() for p in prior})==88,'OAM_PROTECTED_EVIDENCE')
    freeze=ref(plan['historical_freeze'])
    need(plan['historical_freeze']['sha256']==protected['provenance']['freeze_sha256']
         and freeze['identity']['crop_plan_sha256']==protected['provenance']['crop_plan_sha256'],
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
             and contract['seed']==report['seed']
             and report['contract_sha256']==pair['contract']['sha256'],'REAL_A1_BINDING')
        for doc,key in [(contract,'candidates'),(report,'results')]:
            need(len({v['image_id'] for v in doc[key]})==len(doc[key]),'REAL_A1_DUPLICATE')
        refs[role]=(contract,report)
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
        contract,report=refs[role]
        candidates={x['image_id']:x for x in contract['candidates']}
        outcomes={x['image_id']:x for x in report['results']}
        a,c=outcomes[s['image_id']],candidates[s['image_id']]
        reuse=protected_lineage(row,groups[oid],prior)
        need(s['historical_train_reuse'] is reuse and a['final_decision']['status']==qa.ELIGIBLE
             and s['image_id'] in report['eligible_ids'][role] and a['split']==role
             and a['official_split']=='train' and a['oam_id']==c['oam_id']==oid
             and a['validation_fold']==c['validation_fold']==row['validation_fold']
             and a['selected_black_support_cells']==0
             and all(s[k]==a[k] for k in ('crop_xywh','source_hashes','source_hash_verification',
                                           'source_object_sha256','stac_sha256')), 'REAL_A1_SOURCE_BINDING')
        crop=s['crop_xywh']
        need(len(crop)==4 and all(type(v) is int for v in crop) and min(crop[:2])>=0
             and crop[2:]==[224,224] and crop[0]+224<=a['output_shape'][1]
             and crop[1]+224<=a['output_shape'][0],'REAL_CROP')
        status=s['source_hash_verification']; hashes=s['source_hashes']
        need(status in plan['allow_authenticity'] and set(hashes)=={'rgb','annotation'}
             and all(qa.digest(v) for v in hashes.values())
             and (c.get('source_hashes')==hashes if status=='MATCHED_EXTERNAL_EXPECTED_SHA256'
                  else 'source_hashes' not in c),'REAL_AUTHENTICITY_BINDING')
        safe_path(evidence,oid+'-stac.json')
        try:
            verified=qa.source_preflight(row,c,groups,objects,set(),evidence)
        except qa.Decision as exc:
            raise ValueError(exc.result['reason']) from exc
        need(verified['stac_sha256']==s['stac_sha256'],'REAL_STAC_BINDING')
        # Globally pinned A1 separation includes all official opposite-role groups,
        # therefore it also covers the proposed cohort's cross-split comparisons.


def require_physical_control():
    # Logical reservations and free disk are insufficient. No approved quota or
    # whole-attempt supervisor has been demonstrated on the intended pilot volume.
    raise ValueError('REAL_OUTPUT_BLOCKED_PHYSICAL_QUOTA_AND_COORDINATOR_GUARD_UNVERIFIED')


def gate(plan_path, plan_sha, approval_path, approval_sha, root, evidence):
    plan=approval(plan_path,plan_sha,approval_path,approval_sha)
    verify_evidence(plan,plain(plan_path).parent,plain(root),plain(evidence))
    require_physical_control()
