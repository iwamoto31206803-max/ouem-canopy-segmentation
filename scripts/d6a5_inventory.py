"""Rebuild a private D6 source inventory from pinned catalogs and cached evidence."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import time

from d6a3_produce import canonical, need, pinned, plain, write_new
from d6a4 import geographic_order
from d6a32_gate import protected_lineage, SUBINDEX_SHA
import d6a_research_qa as qa
from r1_dataset import safe_path
from r1b_qa_runner import DirectHTTPS

SEED = 118
HOST = 'api.imagery.hotosm.org'
MAX_LOOKUPS = 500
RESPONSE_LIMIT = 128*1024


def read_ref(ref):
    return pinned(plain(Path(ref['path'])),ref['sha256'])


def response(oid, connection=DirectHTTPS):
    """One public metadata GET, without redirects, retries, or asset reads."""
    need(re.fullmatch('[a-f0-9]{24}',oid), 'Invalid public STAC item ID')
    path='/stac/collections/openaerialmap/items/'+oid
    client=connection(HOST,timeout=12)
    try:
        client.request('GET',path,headers={'Accept':'application/json, application/geo+json',
            'Accept-Encoding':'identity','Connection':'close','User-Agent':'OUEM-D6-A5/1'})
        res=client.getresponse()
        kind=(res.getheader('Content-Type') or '').split(';')[0].lower()
        length=res.getheader('Content-Length')
        if res.status!=200:
            return res.status,kind,b'','HTTP_'+str(res.status)
        if (kind not in ('application/json','application/geo+json')
                or res.getheader('Content-Encoding','identity').lower()!='identity'
                or length is not None and (not length.isdecimal() or int(length)>RESPONSE_LIMIT)):
            return res.status,kind,b'','STAC_RESPONSE_FORMAT_OR_LENGTH'
        raw=res.read(RESPONSE_LIMIT+1)
        if len(raw)>RESPONSE_LIMIT or length is not None and int(length)!=len(raw):
            return res.status,kind,raw,'STAC_RESPONSE_LENGTH'
        return res.status,kind,raw,None
    finally:
        client.close()


def collect(order, evidence, log, cap, *, fetch=response, delay=1.0):
    need(type(cap) is int and 0<=cap<=MAX_LOOKUPS, 'STAC_LOOKUP_LIMIT')
    serial=list(order)[:cap]
    need(len({r['oam_id'] for r in serial})==len(serial),'DUPLICATE_LOOKUP')
    failures=0
    with log.open('xb') as dest:
        for row in serial:
            oid=row['oam_id']
            path=evidence/(oid+'-stac.json')
            need(not path.exists(),'STAC_SNAPSHOT_ALREADY_PRESENT')
            entry=dict(oam_id=oid,utc=datetime.now(timezone.utc).isoformat(),body_bytes=0,
                       url=f'https://{HOST}/stac/collections/openaerialmap/items/{oid}')
            try:
                status,kind,raw,error=fetch(oid)
                entry.update(http_status=status,content_type=kind,body_bytes=len(raw))
                if error is None:
                    try:
                        qa.stac_structure(raw,row)
                    except (ValueError,RuntimeError,KeyError,TypeError):
                        error='STAC_IDENTITY_OR_STRUCTURE'
                if error is None:
                    write_new(path,raw,RESPONSE_LIMIT)
                    entry['sha256']=hashlib.sha256(raw).hexdigest()
                entry['status']=error or 'SNAPSHOT_SAVED'
            except (OSError,HTTPException) as exc:
                entry['status']=type(exc).__name__
            dest.write(canonical(entry))
            dest.flush()
            status=entry.get('http_status')
            failures=failures+1 if entry['status']!='SNAPSHOT_SAVED' and status not in (404,410) else 0
            if status==429 or failures>=3:
                return
            if delay:
                time.sleep(delay)


def rebuild(config, evidence):
    """Metadata and filenames only. No TIFF/PNG opening or whole-cache hashing."""
    root=plain(Path(config['cache']))
    paths={k:safe_path(root,v) for k,v in qa.FILES.items() if k!='coco'}
    for k,p in paths.items():
        qa.require_hash(p,config['catalog_sha256'][k])
    rows,groups,objects=qa.load_catalog(paths)
    history=read_ref(config['protected'])
    need(config['protected']['sha256']==SUBINDEX_SHA,'HISTORICAL_ANCHOR')
    parent={s['oam_id']:s for s in read_ref(config['parent_roster'])['sources']}
    report_rows={}
    for ref in config['a1_reports']:
        for a in read_ref(ref)['results']:
            report_rows[a['image_id']]=a['final_decision']['status']
    filenames=root/'extracted/dataset/holdout'
    rgb={p.stem for p in (filenames/'images').glob('*.tif')}
    masks={p.stem for p in (filenames/'masks').glob('*.png')}
    tile_rows=defaultdict(list)
    for row in rows.values():
        tile_rows[row['oam_id']].append(row)
    aliases=Counter(alias for group in groups.values() for alias in group['source_objects'] if alias is not None)
    index=[]
    eligible=[]
    for oid,tiles in sorted(tile_rows.items()):
        group=groups[oid]
        alias=next(iter(group['source_objects'])) if len(group['source_objects'])==1 else None
        snapshot=evidence/(oid+'-stac.json')
        pin=qa.sha(plain(snapshot)) if snapshot.is_file() else None
        state,reason='UNKNOWN','NO_PAIRED_OFFICIAL_TRAIN'
        reuse=None
        pair_count=0
        tile_docs=[]
        representative=None
        bbox_fail=0
        for row in tiles:
            rid=row['image_id']
            stem=f'{oid}_{rid}'
            train=row['official_split']=='train'
            paired=train and stem in rgb and stem in masks
            pair_count+=paired
            tile_docs.append(dict(image_id=rid,fold=row['validation_fold'],role=row['split'],
                alias=qa.identity(row),rgb_present=(stem in rgb) if train else None,
                annotation_present=(stem in masks) if train else None,a1=report_rows.get(rid,'UNKNOWN')))
            if not paired or representative is not None:
                continue
            try:
                reuse=protected_lineage(row,group,history['sources'])
                if pin:
                    qa.source_preflight(row,dict(oam_id=oid,validation_fold=row['validation_fold'],
                                                 stac_sha256=pin),groups,objects,set(),evidence)
                    state,reason='DOCUMENTARY_ELIGIBLE',None
                else:
                    qa.catalog_preflight(row,row,groups,objects,set())
                    state,reason='STAC_MISSING','SNAPSHOT_REQUIRED'
            except qa.Decision as exc:
                state,reason='HOLD',exc.result['reason']
                bbox_fail+=reason=='TILE_OUTSIDE_STAC_DECLARED_BBOX'
                continue
            except (ValueError,TypeError) as exc:
                state,reason='HOLD',str(exc).split(':')[0]
                continue
            representative=row
        if not any(r['official_split']=='train' for r in tiles):
            state,reason='OFFICIAL_TEST_METADATA_ONLY','PROTECTED'
        doc_state=state
        if oid in parent:
            state='MATERIALIZED'
        elif state=='DOCUMENTARY_ELIGIBLE' and any(t['a1']==qa.ELIGIBLE for t in tile_docs):
            state='A1_ELIGIBLE_OBSERVED'
        center=group['center']
        region=[int(x//15) for x in center] if center else None
        index.append(dict(oam_id=oid,roles=sorted(group['roles']),folds=sorted(group['folds']),
            source_aliases=sorted(a for a in group['source_objects'] if a is not None),
            colliding_alias=any(aliases[a]>1 for a in group['source_objects']),
            bounds=group['wgs84_bounds'],region=region,paired_official_train_tiles=pair_count,
            stac_sha256=pin,state=state,documentary_state=doc_state,reason=reason,
            historical_train_reuse=reuse,materialized_image_id=parent.get(oid,{}).get('image_id'),
            stac_bbox_affected_tiles=bbox_fail,tiles=tile_docs))
        if representative is not None and state=='STAC_MISSING':
            eligible.append(dict(representative,region=region,historical_train_reuse=reuse))
    order=[r for role in ('val','train') for r in geographic_order(
        (r for r in eligible if r['split']==role),SEED)]
    return index,order


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--evidence',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--fetch-log',type=Path)
    parser.add_argument('--lookups',type=int,default=0)
    args=parser.parse_args(argv)
    config=pinned(plain(args.config),args.config_sha256)
    evidence=plain(args.evidence)
    need(evidence.is_dir(),'STAC cache directory missing')
    need(not args.lookups or args.fetch_log is not None,'Fetch log required')
    need(not plain(args.out).exists(),'Inventory output already exists')
    index,order=rebuild(config,evidence)
    if args.lookups:
        collect(order,evidence,plain(args.fetch_log),args.lookups)
        index,order=rebuild(config,evidence)
    if args.fetch_log:
        attempts={record['oam_id']:record for record in (
            qa.strict_json(line) for line in plain(args.fetch_log).read_bytes().splitlines())}
        for row in index:
            entry=attempts.get(row['oam_id'])
            if entry:
                row['lookup_status']=entry['status']
                row['lookup_http_status']=entry.get('http_status')
    out=plain(args.out)
    write_new(out,b''.join(map(canonical,index)),8*1024**2)
    print(json.dumps(dict(sha256=qa.sha(out),groups=len(index),states=Counter(r['state'] for r in index),
                          missing_lookup_candidates=len(order))))


if __name__=='__main__':
    main()
