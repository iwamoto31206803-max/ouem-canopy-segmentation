"""D6-A4 cached candidate ordering, guarded A1 reuse and unified shard reader."""
from collections import defaultdict, deque
import json
from pathlib import Path

import d6a_research_qa as qa
from d6a32_gate import PLAN as PILOT, SHARD, ROSTER, EXTENDED_ROSTER, match_roster, protected_lineage, roster_sources
from d6a3_produce import canonical, identifier, need, pinned, plain, write_new
from r1_dataset import safe_path

DATASET = 'd6-sharded-research-v1'
COMPOSED = 'd6-sharded-research-v2'
MAX_CANDIDATES = 300
SEED = 116
SOURCE_FIELDS = ('image_id','oam_id','split','validation_fold','source_object_sha256',
                 'source_hashes','source_hash_verification','historical_train_reuse',
                 'crop_xywh','crop_transform','output_shape','stac_sha256')
QA_LIMITS = dict(commit_bytes=2560*1024**2, rss_bytes=2*1024**3,
                 host_available_bytes=6*1024**3, attempt_seconds=3600)


def geographic_order(rows, seed=SEED):
    """Prefer new sources, then rotate coarse regions in a seeded order."""
    rows=list(rows)
    def key(value):
        return identifier([seed,value])
    ordered=[]
    for reuse in (False, True):
        bins=defaultdict(list)
        for row in rows:
            if row['historical_train_reuse'] is reuse:
                bins[tuple(row['region'])].append(row)
        queues=[deque(sorted(bins[cell], key=lambda row:key(row['image_id'])))
                for cell in sorted(bins,key=key)]
        while queues:
            for queue in queues:
                ordered.append(queue.popleft())
            queues=[queue for queue in queues if queue]
    return ordered


def validate_roster_catalog(roster, rows, groups, prior):
    """Check the shared roster, including the reused pilot, against pinned scope."""
    for source in roster_sources(roster):
        row=rows[source['image_id']]
        oid=row['oam_id']
        need(row['official_split']=='train'
             and all(source[k]==row[k] for k in ('oam_id','split','validation_fold'))
             and source['source_object_sha256']==qa.identity(row)
             and source['historical_train_reuse'] is protected_lineage(row,groups[oid],prior),
             'D6_ROSTER_CATALOG_BINDING')


def qa_worker(root, evidence, contract, pin, output):
    """Run existing A1 under the existing Windows Job. Output is report only."""
    from d6a3_guard import require_local_fixed
    from d6a3_produce import offline_worker, output_scope
    from r1d_bounded_prepare import WindowsJob

    offline_worker()
    WindowsJob.require_current(QA_LIMITS['commit_bytes'])
    root,evidence,contract,output=map(Path,(root,evidence,contract,output))
    require_local_fixed(root,evidence,contract,output)
    root,evidence,contract,output=map(plain,(root,evidence,contract,output))
    output_scope(contract,root,evidence,output)
    plan=pinned(contract,pin)
    need(0<len(plan['candidates'])<=MAX_CANDIDATES and not output.exists(), 'D6_QA_SCOPE')
    report=qa.assess(root,evidence,contract,pin)
    write_new(output,canonical(report),8*1024**2)
    print(json.dumps(dict(report_sha256=qa.sha(output),counts=report['counts'])))


class D6Shards:
    """A frozen dataset-level index over existing D6 generations."""
    def __init__(self, manifest, manifest_sha256, *, _allow_parent=True):
        from d6a3_data import D6Data

        manifest=plain(Path(manifest))
        self._manifest=manifest
        doc=pinned(manifest,manifest_sha256)
        body={k:v for k,v in doc.items() if k!='generation_id'}
        need((doc['schema']==DATASET or _allow_parent and doc['schema']==COMPOSED) and doc['generation_id']==identifier(body), 'D6_AGGREGATE_IDENTITY')
        roster_ref=doc['roster']
        roster=pinned(safe_path(manifest.parent,roster_ref['path']),roster_ref['sha256'])
        composed=doc['schema']==COMPOSED
        need(roster['schema']==(EXTENDED_ROSTER if composed else ROSTER),'D6_AGGREGATE_ROSTER_SCHEMA')
        expected=roster_sources(roster)
        self.sources=expected
        self.samples,self.crop_positions,self._readers={}, {}, {}
        self.records={}
        self.identity=dict(dataset_id=doc['generation_id'], manifest_sha256=manifest_sha256,
                           purpose='EXPLORATORY_RESEARCH_ONLY',source_generation=qa.GENERATION)
        records=[]
        patch_ids=set()
        shard_roots=set()
        if composed:
            ref=doc['parent']
            need(ref['sha256']==roster['parent_sha256'],'D6_PARENT_PIN')
            parent=D6Shards(ref['path'],ref['sha256'],_allow_parent=False)
            count=roster['parent_count']
            need(parent.complete and type(count) is int and len(parent.samples)==count
                 and parent.sources==expected[:count],'D6_PARENT_SOURCE_SCOPE')
            self.samples.update(parent.samples)
            self.crop_positions.update(parent.crop_positions)
            self._readers.update(parent._readers)
            self.records.update(parent.records)
            records.extend(parent.samples)
            patch_ids.update(r['patch_id'] for r in parent.records.values())
        for i,ref in enumerate(doc['shards']):
            root=plain(Path(ref['path']))
            need(root not in shard_roots, 'D6_DUPLICATE_SHARD')
            shard_roots.add(root)
            reader=D6Data(root,ref['sha256'])
            plan=reader.generation_plan
            if i==0 and not composed:
                need(plan['schema']==PILOT and ref['sha256']==roster['pilot_sha256']
                     and len(reader.samples)==4, 'D6_PILOT_SHARD_BINDING')
            else:
                need(plan['schema']==SHARD and plan['roster']['sha256']==roster_ref['sha256']
                     and plan['shard_start']==len(records), 'D6_SHARD_ORDER_OR_ROSTER')
                match_roster(plan,roster)
            need(plan['sources']==expected[len(records):len(records)+len(reader.samples)],
                 'D6_AGGREGATE_SOURCE_SCOPE')
            for rid,sample in reader.samples.items():
                receipt,_,_=reader._records[rid]
                patch_id=receipt['patch_id']
                need(rid not in self.samples and patch_id not in patch_ids, 'D6_AGGREGATE_DUPLICATE')
                self.samples[rid]=sample
                self.crop_positions[rid]=reader.crop_positions[rid]
                self._readers[rid]=reader
                self.records[rid]=receipt
                patch_ids.add(patch_id)
                records.append(rid)
        roles={role:[rid for rid in records if self.samples[rid]['split']==role] for role in ('train','val')}
        need(roles==doc['eligible_ids'] and all(roles.values()) and type(doc['complete']) is bool
             and doc['complete']==(len(records)==len(expected)), 'D6_AGGREGATE_ALLOWLIST')
        self.complete=doc['complete']

    def patch(self,item,split):
        rid=item['image_id']
        need(type(rid) is int and rid in self._readers,'UNKNOWN_D6_SOURCE')
        qa.require_hash(self._manifest,self.identity['manifest_sha256'])
        return self._readers[rid].patch(item,split)


def aggregate_document(roster_ref, roster, shards, completed, *, parent=None):
    sources=roster_sources(roster)
    need(type(completed) is int and 0<completed<=len(sources), 'D6_AGGREGATE_SCOPE')
    doc=dict(schema=DATASET,roster=roster_ref,shards=shards,complete=(completed==len(sources)),
             eligible_ids={role:[s['image_id'] for s in sources[:completed] if s['split']==role]
                           for role in ('train','val')})
    if parent is not None:
        need(roster['schema']==EXTENDED_ROSTER and parent['sha256']==roster['parent_sha256'],
             'D6_PARENT_PIN')
        doc.update(schema=COMPOSED,parent=parent)
    return dict(doc,generation_id=identifier(doc))

def select_cached(rows, groups, objects, prior, evidence, root, pilot, prior_results, *, seed=SEED, limit=MAX_CANDIDATES, skip_ids=()):
    """Pick at most one documentary candidate per source, before any pixel QA."""
    from collections import Counter
    from d6a3_produce import source_paths

    blocked_oams={s['oam_id'].casefold() for s in pilot}
    blocked_aliases={s['source_object_sha256'] for s in pilot}
    sources=defaultdict(list)
    for row in rows.values():
        if row['official_split']=='train':
            sources[row['oam_id']].append(row)
    chosen=[]
    holds=Counter()
    for item in sorted(evidence.glob('*-stac.json')):
        oid=item.name.removesuffix('-stac.json')
        if oid.casefold() in blocked_oams:
            holds['PILOT_REUSED']+=1
            continue
        tiles=sources.get(oid, [])
        if not tiles:
            holds['NO_OFFICIAL_TRAIN']+=1
            continue
        reason='NO_DOCUMENTARY_TILE'
        snapshot_sha=qa.sha(item)
        for row in sorted(tiles,key=lambda r:(r['image_id'] not in prior_results,
                                             identifier([seed,r['image_id']]))):
            rid=row['image_id']
            if rid in skip_ids:
                reason='PRIOR_A1_HOLD'
                continue
            candidate={k:row[k] for k in ('image_id','oam_id','validation_fold')}
            candidate['stac_sha256']=snapshot_sha
            a=prior_results.get(rid)
            if a is not None and a['stac_sha256']==snapshot_sha and 'expected_source_hashes' in a:
                candidate['source_hashes']=a['expected_source_hashes']
            group=groups[oid]
            try:
                declared=qa.source_preflight(row,candidate,groups,objects,set(),evidence)
                reuse=protected_lineage(row,group,prior)
                source_paths(root,row)
            except qa.Decision as exc:
                reason=exc.result['reason']
                continue
            except ValueError as exc:
                reason=str(exc).split(':')[0]
                continue
            alias=declared['source_object_sha256']
            if alias in blocked_aliases:
                reason='DUPLICATE_METADATA_OBJECT'
                break
            blocked_aliases.add(alias)
            blocked_oams.add(oid.casefold())
            center=group['center']
            chosen.append(dict(candidate=candidate,split=row['split'],image_id=rid, region=[int(x//15) for x in center],
                source_object_sha256=alias,historical_train_reuse=reuse,
                reusable_a1=(a is not None and a['stac_sha256']==snapshot_sha
                             and all(a[k]==v for k,v in declared.items()))))
            break
        else:
            holds[reason]+=1
            continue
        if reason=='DUPLICATE_METADATA_OBJECT':
            holds[reason]+=1
    ordered={role:geographic_order((r for r in chosen if r['split']==role),seed)
             for role in ('train','val')}
    if sum(map(len,ordered.values()))>limit:
        ordered['val']=ordered['val'][:limit]
        ordered['train']=ordered['train'][:limit-len(ordered['val'])]
        holds['CANDIDATE_BUDGET_REMAINDER']=len(chosen)-limit
    return ordered, dict(holds)
