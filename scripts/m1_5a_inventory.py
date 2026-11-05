"""Revision-pinned metadata-only inventory and provisional expansion selection."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer,Geod
import pyarrow.parquet as pq
from m1_2a_data_qa import Budget as BaseBudget,RangeFile,REVISION,CODE_REVISION,ROOT


class Budget(BaseBudget):
    """Reuse TLS connections and publisher redirects during bounded range reads."""
    def __init__(self, limit):
        super().__init__(limit)
        self.session = requests.Session()
        self.resolved = {}

    def get(self, url, headers=None):
        with self.session.get(self.resolved.get(url, url), headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            if headers and response.status_code != 206:
                raise ValueError("Server did not honor range")
            self.resolved[url] = response.url
            chunks = []
            for chunk in response.iter_content(65536):
                self.used += len(chunk)
                if self.used > self.limit:
                    raise ValueError("Transfer budget exceeded")
                chunks.append(chunk)
            return b"".join(chunks)

COLUMNS=['image_id','oam_id','license','validation_fold','height','width','crs','bounds','biome','meta']
EXISTING={661:'train',1972:'val'}


def role(row):
    if row['official_split']=='test': return 'holdout'
    fold=row.get('validation_fold')
    return 'val' if fold==0 else 'train' if fold in (1,2,3,4) else 'hold'


def eligibility(row):
    required=['image_id','oam_id','license','validation_fold','height','width','crs','bounds','meta']
    if any(row.get(k) is None for k in required): return 'HOLD_MISSING_METADATA'
    if row['license']!='CC-BY 4.0': return 'HOLD_LICENSE'
    if role(row) not in ('train','val'): return 'HOLD_SPLIT'
    if not row.get('wgs84_bounds'): return 'HOLD_GEOMETRY'
    return 'PROVISIONAL_SOURCE_DECLARED'


def split_audit(rows):
    groups=defaultdict(set); folds=defaultdict(set); seen=set()
    for row in rows:
        key=row['image_id']
        if key in seen: raise ValueError('Duplicate image_id')
        seen.add(key)
        groups[row['oam_id']].add(role(row))
        folds[row['oam_id']].add((row['official_split'],row['validation_fold']))
        if key in EXISTING and role(row)!=EXISTING[key]: raise ValueError('Existing role conflict')
    conflicts={k:sorted(v) for k,v in groups.items() if len(v)>1}
    fold_conflicts=[k for k,v in folds.items() if len(v)>1]
    if conflicts or fold_conflicts: raise ValueError('oam_id crosses role/fold/holdout')
    for key in EXISTING:
        if key not in seen: raise ValueError('Existing sample missing')
    return dict(unique_oam_ids=len(groups),duplicate_oam_groups=sum(sum(r['oam_id']==g for r in rows)>1 for g in groups),
                cross_role_conflicts=conflicts,cross_fold_conflicts=fold_conflicts,existing_roles_preserved=True)


@lru_cache(maxsize=16)
def inverse_transform(crs):
    return Transformer.from_crs(crs,4326,always_xy=True)


def geometry(row):
    left,bottom,right,top=row['bounds']
    if not (right>left and top>bottom and row['width']>0 and row['height']>0): raise ValueError('Invalid bounds')
    inverse=inverse_transform(row['crs'])
    x,y=(left+right)/2,(bottom+top)/2
    lon,lat=inverse.transform(x,y)
    if not np.isfinite([lon,lat]).all() or abs(lat)>90 or abs(lon)>180: raise ValueError('Invalid location')
    dx,dy=(right-left)/row['width'],(top-bottom)/row['height']
    geod=Geod(ellps='WGS84')
    a=inverse.transform(left,bottom); b=inverse.transform(right,top)
    return dict(center=[lon,lat],wgs84_bounds=[*a,*b],projected_pixel=[dx,dy],
                estimated_ground_pixel_m=[geod.inv(lon,lat,*inverse.transform(x+dx,y))[2],
                                          geod.inv(lon,lat,*inverse.transform(x,y+dy))[2]])


def overlaps(a,b):
    return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])


def select_candidates(rows,train_count=8,val_count=4):
    blocked={r['oam_id'] for r in rows if r['image_id'] in EXISTING}
    selected=[]; cells=set(); biomes=set(); reasons={}
    remaining=sorted(rows,key=lambda r:r['image_id'])
    opposites={desired:[r for r in rows if role(r)!=desired and r.get('wgs84_bounds')] for desired in ['train','val']}
    for desired,count in [('train',train_count),('val',val_count)]:
        for _ in range(count):
            pool=[]
            for row in remaining:
                rid=row['image_id']; state=eligibility(row)
                if state!='PROVISIONAL_SOURCE_DECLARED': reasons[rid]=state; continue
                if role(row)!=desired: continue
                if row['oam_id'] in blocked: reasons[rid]='EXCLUDED_EXISTING_OR_SELECTED_OAM'; continue
                opposite=opposites[desired]
                if any(overlaps(row['wgs84_bounds'],r['wgs84_bounds']) for r in opposite):
                    reasons[rid]='HOLD_CROSS_ROLE_FOOTPRINT_OVERLAP'; continue
                lon,lat=row['center']; cell=(int(np.floor(lon/10)),int(np.floor(lat/10)))
                score=(cell in cells,row['biome'] in biomes,row['image_id'])
                pool.append((score,row,cell))
            if not pool: break
            _,chosen,cell=min(pool,key=lambda x:x[0])
            chosen=dict(chosen,assigned_role=desired,selection_reason='New oam; nonoverlapping opposite-role bbox; prefer unused 10deg cell, then biome, then lowest image_id')
            selected.append(chosen); blocked.add(chosen['oam_id']); cells.add(cell); biomes.add(chosen['biome'])
            remaining=[r for r in remaining if r['image_id']!=chosen['image_id']]
            reasons[chosen['image_id']]='SELECTED_PROVISIONAL'
    return selected,reasons



def verify_existing(rows, established):
    lookup = {r['image_id']:r for r in rows}
    for sample in established:
        row = lookup.get(sample['image_id'])
        if row is None or any(row[k] != sample[k] for k in ['oam_id','license','validation_fold']):
            raise ValueError('Existing sample identity/license/fold conflict')
        if role(row) != sample['split']:
            raise ValueError('Existing assigned role conflict')
    return True


def distributions(rows):
    located = [r for r in rows if r.get('center')]
    pixels = [v for r in located for v in r['estimated_ground_pixel_m']]
    return dict(geometry_unverified_rows=len(rows)-len(located),
        longitude_range=[min(r['center'][0] for r in located),max(r['center'][0] for r in located)],
        latitude_range=[min(r['center'][1] for r in located),max(r['center'][1] for r in located)],
        coarse_cell_counts=dict(Counter(str([int(np.floor(r['center'][0]/10)),int(np.floor(r['center'][1]/10))]) for r in located)),
        estimated_ground_pixel_m_range=[min(pixels),max(pixels)],
        estimated_ground_pixel_m_quantiles=np.quantile(pixels,[0,.25,.5,.75,1]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.cache.resolve()==repo or repo in args.cache.resolve().parents: parser.error('External metadata cache required')
    budget=Budget(128*1024**2)
    if args.offline:
        rows=json.loads((args.cache/'rows.json').read_text())
        acquisition=json.loads((args.cache/'acquisition.json').read_text())
    else:
        if args.cache.exists(): parser.error('Use fresh metadata cache')
        args.cache.mkdir(parents=True)
        info=json.loads(budget.get(f'https://huggingface.co/api/datasets/restor/tcd/revision/{REVISION}?blobs=true'))
        if info['sha']!=REVISION: raise ValueError('Revision mismatch')
        rows=[]; shards=[]
        for item in sorted(info['siblings'],key=lambda s:s['rfilename']):
            name=item['rfilename']
            if not name.startswith(('data/train-','data/test-')): continue
            remote=RangeFile(ROOT+'/'+name,item['size'],budget)
            table=pq.ParquetFile(remote).read(columns=COLUMNS)
            print(json.dumps(dict(shard=name, rows=table.num_rows, payload_bytes=budget.used)),flush=True)
            split='test' if name.startswith('data/test-') else 'train'
            for index,row in enumerate(table.to_pylist()):
                row.update(official_split=split,shard=name,shard_row=index)
                try: row.update(geometry(row))
                except Exception as error: row['geometry_error']=type(error).__name__
                rows.append(row)
            shards.append(dict(path=name,publisher_sha256=item.get('lfs',{}).get('sha256'),rows=table.num_rows))
        acquisition=dict(utc=datetime.now(timezone.utc).isoformat(),revision=REVISION,metadata_columns=COLUMNS,
                         transfer_payload_bytes=budget.used,shards=shards)
        (args.cache/'rows.json').write_text(json.dumps(rows))
        (args.cache/'acquisition.json').write_text(json.dumps(acquisition,indent=2))
    established=json.loads(Path('docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())['samples']
    existing_verified=verify_existing(rows, established)
    splits=split_audit(rows)
    selected,reasons=select_candidates(rows)
    # Only selected original-source metadata is requested; no images or annotations.
    evidence_path=args.cache/'candidate_evidence.json'
    if args.offline: evidence=json.loads(evidence_path.read_text())
    else:
        evidence={}
        for row in selected:
            url='https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+row['oam_id']
            entry=dict(url=url,status='HOLD_SOURCE_UNVERIFIED')
            try:
                raw=budget.get(url); item=json.loads(raw)
                (args.cache/(row['oam_id']+'-stac.json')).write_bytes(raw)
                providers=item['properties'].get('providers',[])
                consistent=(item['id']==row['oam_id'] and item['properties'].get('license','').lower()=='cc-by-4.0')
                entry.update(status='SOURCE_LICENSE_VERIFIED_PIXEL_QA_PENDING' if consistent else 'HOLD_SOURCE_CONFLICT',
                    source_license=item['properties'].get('license'),attribution_provider_present=bool(providers),
                    snapshot_sha256=hashlib.sha256(raw).hexdigest())
            except Exception as error: entry['error']=type(error).__name__
            evidence[str(row['image_id'])]=entry
        evidence_path.write_text(json.dumps(evidence,indent=2))
    geod=Geod(ellps='WGS84')
    for row in selected:
        entry=evidence[str(row['image_id'])]
        if entry.get('source_license') is None and entry.get('snapshot_sha256'):
            entry['status']='HOLD_SOURCE_LICENSE_MISSING'
        snapshot=args.cache/(row['oam_id']+'-stac.json')
        if entry.get('snapshot_sha256'):
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=entry['snapshot_sha256']:
                raise ValueError('Cached STAC snapshot hash mismatch')
        opposite=[r for r in rows if role(r)!=role(row) and r.get('center')]
        lon,lat=row['center']
        row['nearest_opposite_role_center_m']=min(geod.inv(lon,lat,*r['center'])[2] for r in opposite)
        row['m1_5b_disposition']='PROVISIONAL_ACQUIRE_FOR_QA' if entry['status']=='SOURCE_LICENSE_VERIFIED_PIXEL_QA_PENDING' and entry.get('attribution_provider_present') else 'HOLD'
        row['source_evidence']=evidence[str(row['image_id'])]
        row['training_eligible']=False
        row['category_pixel_qa']='NOT_VERIFIED'
    result=dict(revision=REVISION,official_code_revision=CODE_REVISION,acquisition=acquisition,
                existing_sample_identity_verified=existing_verified,geography=distributions(rows),
                counts=dict(total=len(rows),official_split=dict(Counter(r['official_split'] for r in rows)),
                    roles=dict(Counter(role(r) for r in rows)),licenses=dict(Counter(r['license'] for r in rows)),
                    folds=dict(Counter(str(r['validation_fold']) for r in rows)),crs=dict(Counter(r['crs'] for r in rows)),
                    shapes=dict(Counter(str([r['height'],r['width']]) for r in rows))),split_audit=splits,
                role_oam_counts={s:len({r['oam_id'] for r in rows if role(r)==s}) for s in ['train','val','holdout']},
                candidates=selected,candidate_reason_counts=dict(Counter(reasons.get(r['image_id'],'NOT_SELECTED_WITHIN_BUDGET') for r in rows)),
                metadata_cache_sha256=hashlib.sha256((args.cache/'rows.json').read_bytes()).hexdigest(),
                status='METADATA_AUDIT_COMPLETE_CANDIDATES_PROVISIONAL',
                total_network_payload_bytes=budget.used if not args.offline else json.loads((args.cache/'result.json').read_text())['total_network_payload_bytes'],
                offline_replay=args.offline,
                limitations=['No image or annotation column read','Metadata bounds are float32, embedded grids unverified',
                             'Canopy coverage/quality and raw pixel IDs unverified for new candidates'])
    (args.cache/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    # Per-row disposition contains identifiers and reasons only.
    (args.cache/'dispositions.json').write_text(json.dumps([dict(image_id=r['image_id'],role=role(r),
        reason=reasons.get(r['image_id'],'NOT_SELECTED_WITHIN_BUDGET')) for r in rows],indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
