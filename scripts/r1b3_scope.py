"""Versioned, offline 150-minus-25 scope; shares the existing bounded QA ledger."""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

import r1b_campaign as ledger
import r1b_qa_runner as qa
from m1_5b2_border_audit import component_report
from m1_5d_zenodo_prepare import local_index
from r1b_candidate_plan import identity

SCHEMA='ouem-r1b3-scope-v1'
CAMPAIGN_SCHEMA='ouem-r1b3-local-campaign-v1'
LIMITS=dict(network_bytes=0,requests=0,response_bytes=65536,storage_bytes=256*1024**2)


def remaining(plan):
    """Exact ordered remainder, never a new selection or an eligibility decision."""
    pool=plan['plan']['pool'];old=plan['plan']['initial_batch_ids']
    if len(pool)!=150 or len(old)!=25 or [r['image_id'] for r in pool[:25]]!=old:
        raise ledger.Stop('Scope requires fixed complete 150 and initial 25')
    if len({r['image_id'] for r in pool})!=150:raise ledger.Stop('Duplicate pool image ID')
    roles=[]
    for r in pool:
        if (type(r['image_id']) is not int or not isinstance(r['oam_id'],str) or
                not re.fullmatch('[A-Za-z0-9-]+',r['oam_id']) or r['official_split']!='train' or
                type(r['validation_fold']) is not int or r['validation_fold'] not in range(5) or
                r['split']!=('val' if r['validation_fold']==0 else 'train') or r['row_license']!='CC-BY 4.0' or
                not isinstance(r['original_source_identity_sha256'],str) or
                not re.fullmatch('[0-9a-f]{64}',r['original_source_identity_sha256'])):
            raise ledger.Stop('Invalid pool identity/role/license')
        roles.append(r['split'])
    if roles!=['train','train','train','train','val']*30:raise ledger.Stop('Fixed pool split/order mismatch')
    if len({r['oam_id'] for r in pool})!=150 or len({r['original_source_identity_sha256'] for r in pool})!=150:
        raise ledger.Stop('Duplicate pool source')
    return pool[25:]


def selected_records(plan,cache):
    selected=remaining(plan);rows=local_index(cache);indexed={r['image_id']:r for r in rows}
    object_roles={}
    for r in rows:
        object_roles.setdefault(identity(r),set()).add((r['official_split'],r['validation_fold']))
    # Check the full pool, including reused records, against Train metadata only.
    for p in plan['plan']['pool']:
        r=indexed[p['image_id']]
        if (r['official_split']!='train' or r['oam_id']!=p['oam_id'] or
                r['validation_fold']!=p['validation_fold'] or r['license']!=p['row_license'] or
                identity(r)!=p['original_source_identity_sha256'] or
                len(object_roles[identity(r)])!=1 or
                [r['height'],r['width']]!=p['shape'] or
                r['wgs84_bounds']!=p['representative_metadata_bounds_wgs84']):
            raise ledger.Stop('Pool/source metadata mismatch')
    records=[]
    for r in selected:
        stem=f"{r['oam_id']}_{r['image_id']}"
        files=dict(rgb=f'extracted/dataset/holdout/images/{stem}.tif',
                   annotation=f'extracted/dataset/holdout/masks/{stem}.png',
                   byte_validation='NOT_VERIFIED_R1B2B_REQUIRED')
        for key in ('rgb','annotation'):
            path=cache/files[key];ledger.ordinary(path);path.resolve().relative_to(cache.resolve())
            if not path.is_file():raise ledger.Stop('Missing scoped source file')
        records.append(dict(image_id=r['image_id'],oam_id=r['oam_id'],split=r['split'],
            official_split='train',validation_fold=r['validation_fold'],source_files=files))
    return records


def reuse_evidence(parent_path,parent_sha,cache,legacy_root,legacy_registry):
    """Read committed old reports; never create imported SUCCESS ledger operations."""
    view=ledger.Campaign.inspect(legacy_root,legacy_registry,parent_path,parent_sha)
    if view['status']!='CLEAN_PAUSED' or view['contract']['cache_root']!=str(cache.resolve()):
        raise ledger.Stop('Legacy evidence not clean or wrong cache')
    contract=view['contract'];records=[]
    for key,split in contract['targets'].items():
        rid=int(key);refs={}
        for kind in ('report','preview'):
            op=view['operations'].get(f'qa-{kind}-{rid}',{})
            if op.get('status')!='SUCCESS':raise ledger.Stop('Missing committed legacy evidence')
            refs[kind]=dict(path=op['output_path'],sha256=op['file']['sha256'])
        report=qa.strict_json((Path(legacy_root)/refs['report']['path']).read_bytes())
        if (report.get('image_id')!=rid or report.get('split')!=split or
                report.get('source_generation')!=ledger.GENERATION or
                report.get('technical_status') not in ('PASS','HOLD','FAIL')):
            raise ledger.Stop('Legacy report identity mismatch')
        source=report['checks']['source_hashes']
        hashes={}
        for kind,hkey in [('rgb','image_sha256'),('annotation','annotation_sha256')]:
            path=cache/contract['source_files'][key][kind];ledger.ordinary(path)
            if ledger.sha(path)!=source[hkey]:raise ledger.Stop('Legacy source SHA mismatch')
            hashes[hkey]=source[hkey]
        records.append(dict(image_id=rid,split=split,source_hashes=hashes,**refs))
    return dict(root=str(Path(legacy_root).resolve()),registry=str(Path(legacy_registry).resolve()),
        contract_sha256=ledger.sha(Path(legacy_root)/'contract.json'),samples=records,
        r1_acceptance='NOT_INHERITED')


def build_scope(parent_path,parent_sha,cache,legacy_root,legacy_registry,expected_parent_id=None):
    """Prepare a reviewable scope object in memory. No pixels, HTTP or writes."""
    parent_path=Path(parent_path);cache=Path(cache)
    parent,_,metadata=ledger.inputs(parent_path,parent_sha,cache,expected_parent_id)
    records=selected_records(parent,cache)
    scope=dict(schema=SCHEMA,source_generation=ledger.GENERATION,cache_root=str(cache.resolve()),
        parent_plan=dict(path=str(parent_path.resolve()),plan_id=parent['plan_id'],sha256=parent_sha),
        metadata_sha256=metadata,samples=records,
        reused_evidence=reuse_evidence(parent_path,parent_sha,cache,legacy_root,legacy_registry))
    scope['scope_id']='ouem-r1b3-'+ledger.hash_bytes(ledger.encoded(scope))
    return scope


def scope_inputs(path,expected_sha,cache,expected_id=None):
    path=Path(path);cache=Path(cache);ledger.ordinary(path)
    if ledger.sha(path)!=expected_sha:raise ledger.Stop('Scope file SHA mismatch')
    scope=qa.strict_json(path.read_bytes());core={k:v for k,v in scope.items() if k!='scope_id'}
    if scope.get('schema')!=SCHEMA or scope.get('source_generation')!=ledger.GENERATION or scope.get('scope_id')!='ouem-r1b3-'+ledger.hash_bytes(ledger.encoded(core)):
        raise ledger.Stop('Scope ID/schema/generation mismatch')
    if expected_id is not None and scope['scope_id']!=expected_id:raise ledger.Stop('Scope ID mismatch')
    parent=scope['parent_plan'];reuse=scope['reused_evidence']
    actual=build_scope(parent['path'],parent['sha256'],cache,reuse['root'],reuse['registry'],parent['plan_id'])
    if ledger.encoded(actual)!=ledger.encoded(scope):raise ledger.Stop('Scope records/evidence mismatch')
    records=scope['samples'];ids=[r['image_id'] for r in records]
    # Adapter returns the narrow input shape expected by the existing ledger/Runner.
    proxy=dict(plan_id=scope['scope_id'],plan=dict(initial_batch_ids=ids,pool=records),
               _scope=scope,_scope_path=str(path.resolve()),_scope_sha256=expected_sha)
    return proxy,{str(r['image_id']):r['split'] for r in records},scope['metadata_sha256']


def coverage_diagnostics(rgb,target,black):
    """Review flags only. No NoData decision, validity mask or technical override."""
    white,_,_=component_report(np.all(rgb==255,axis=2),target)
    h,w=target.shape;colors={};flags=[];shapes=[]
    for color,data in [('black',black),('white',white)]:
        colors[color]=dict(pixels=data['black_pixels'],border_connected_pixels=data['border_connected_pixels'],
            border_positive_label_pixels=data['border_positive_label_pixels'],largest=data['largest'],connectivity=4)
        if data['border_connected_pixels']:flags.append('EDGE_CONNECTED_'+color.upper())
        for comp in data['largest']:
            x0,y0,x1,y1=comp['bbox_xyxy']
            if comp['touches_border'] and (x0==0 or x1==w) and (y0==0 or y1==h):
                shapes.append(dict(color=color,reason='POSSIBLE_CORNER_OR_WEDGE_REVIEW',**comp))
    return dict(status='REVIEW_REQUIRED' if flags else 'NOT_VERIFIED',flags=flags,colors=colors,
        shape_candidates=shapes,no_data_classification='NOT_DETERMINED',valid_mask_generated=False,
        shape_scope='largest five components per color; edge totals include all components',
        policy='diagnostic flags only; no new numeric acceptance threshold; technical status unchanged')


class ScopeCampaign(ledger.Campaign):
    schema=CAMPAIGN_SCHEMA
    read_inputs=staticmethod(scope_inputs)
    coverage_diagnostics=staticmethod(coverage_diagnostics)

    @classmethod
    def checked_limits(cls,limits):
        limits=dict(LIMITS if limits is None else limits)
        if (set(limits)!=set(LIMITS) or any(type(v) is not int for v in limits.values()) or
                limits['network_bytes']!=0 or limits['requests']!=0 or
                not 0<limits['response_bytes']<=65536 or not 0<limits['storage_bytes']<=LIMITS['storage_bytes']):
            raise ledger.Stop('Scope requires zero network/requests and storage <=256MiB')
        return limits

    @staticmethod
    def contract_metadata(plan):
        scope=plan['_scope'];ids=plan['plan']['initial_batch_ids']
        return dict(scope_id=scope['scope_id'],scope_sha256=plan['_scope_sha256'],
            parent_plan=scope['parent_plan'],reused_evidence=scope['reused_evidence'],
            batches=[ids[i:i+25] for i in range(0,125,25)])

    @staticmethod
    def protected_inputs(plan):
        s=plan['_scope']
        return [plan['_scope_path'],s['parent_plan']['path'],s['reused_evidence']['root'],s['reused_evidence']['registry']]

    @staticmethod
    def validate_approval(record,contract):
        allowed=set(record['allowed_image_ids'])
        if (not set(record['scopes'])<= {'qa_local','qa_preview','qa_report'} or len(allowed)>25 or
                not any(allowed<=set(batch) for batch in contract['batches'])):
            raise ledger.Stop('Scope approval must be local-only within one fixed batch of 25')
        if record.get('scope_id')!=contract['scope_id'] or record.get('scope_sha256')!=contract['scope_sha256']:
            raise ledger.Stop('Approval scope binding mismatch')

    def _load(self,*args,**kwargs):
        super()._load(*args,**kwargs)
        self.checked_limits(self.contract['limits'])

    def reserve(self,*args,**kwargs):
        # The shared ledger enforces zero bytes/requests; this path also bans GeoTIFF.
        kind=kwargs.get('kind',args[3] if len(args)>3 else None)
        if kind in ('stac','geotiff'):raise ledger.Stop('Local QA scope forbids STAC/GeoTIFF operations')
        return super().reserve(*args,**kwargs)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h','--help'):
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action',choices=['build-scope','init','inspect','resume-check','reconcile','check-approval','run'])
        parser.print_help()
        return 0 if args else 2
    if args and args[0]=='build-scope':
        parser=argparse.ArgumentParser(description='Prepare scope only; no campaign/approval/QA')
        for key in ('parent-plan','cache','legacy-root','legacy-registry','output'):
            parser.add_argument('--'+key,type=Path,required=True)
        parser.add_argument('--parent-sha256',required=True);parser.add_argument('--parent-plan-id',required=True)
        a=parser.parse_args(args[1:])
        scope=build_scope(a.parent_plan,a.parent_sha256,a.cache,a.legacy_root,a.legacy_registry,a.parent_plan_id)
        ledger.ordinary(a.output)
        # Scope is outside all protected inputs, no parent directories are created.
        for source in [a.cache,a.legacy_root,a.legacy_registry,Path(__file__).resolve().parents[1]]:
            if a.output.resolve()==source.resolve() or source.resolve() in a.output.resolve().parents:
                raise ledger.Stop('Scope output overlaps protected source')
        ledger.durable(a.output,ledger.encoded(scope))
        print(json.dumps(dict(scope_id=scope['scope_id'],scope_sha256=ledger.sha(a.output),qa_executed=False)))
        return 0
    if args and args[0] in ('check-approval','run'):
        return qa.main(args,campaign_type=ScopeCampaign)
    return ledger.main(args,campaign_type=ScopeCampaign)


if __name__=='__main__':raise SystemExit(main())
