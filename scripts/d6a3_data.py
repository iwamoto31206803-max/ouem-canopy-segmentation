"""Reader for immutable D6 research generations, separate from Formal R1."""
from pathlib import Path

import torch

from d6a3_produce import (AUTH, PLAN, RECEIPT, SCHEMA, canonical, identifier, inventory, need,
                          pinned, plain, validate_source)
from d6a_research_qa import ELIGIBLE, GENERATION, digest
from r1_dataset import safe_path


class D6Data:
    def __init__(self, root, manifest_sha256):
        self.root = plain(Path(root))
        self.manifest_sha256 = manifest_sha256
        manifest = pinned(safe_path(self.root, 'manifest.json'), manifest_sha256)
        need(set(inventory(self.root)) == {'plan.json','sources','commits','manifest.json'}
             and manifest['schema'] == SCHEMA, 'GENERATION_NOT_FROZEN')
        body = {k:v for k,v in manifest.items() if k != 'generation_id'}
        need(identifier(body) == manifest['generation_id'], 'GENERATION_ID_MISMATCH')
        plan = pinned(safe_path(self.root,'plan.json'), manifest['plan_sha256'])
        from d6a32_gate import GT54, PLAN as REAL_PLAN, SHARD, MAX_SHARD_SOURCES, STORAGE_POLICY
        shard = plan['schema'] == SHARD
        real = shard or plan['schema'] == REAL_PLAN
        need(plan['windows_per_source'] == {'train':1,'val':1} and (
             (not real and plan['schema'] == PLAN and plan['mode'] == 'synthetic' and 'mode' not in manifest)
             or (real and manifest['mode']==('real-shard' if shard else 'real-pilot')
                 and manifest['source_generation']==plan['source_generation']==GENERATION
                 and plan['storage_policy']==STORAGE_POLICY and plan['gt54']==GT54)), 'D6_PLAN_BINDING')
        planned = {s['image_id']:s for s in plan['sources']} if real else {}
        if real:
            need(len(planned)==len(plan['sources']) and 0<len(planned)<=(MAX_SHARD_SOURCES if shard else 4), 'D6_PLAN_SCOPE')
        self.generation_plan = plan
        self.identity = dict(dataset_id=manifest['generation_id'], manifest_sha256=manifest_sha256,
                             source_generation=GENERATION, purpose='EXPLORATORY_RESEARCH_ONLY')
        refs = manifest['source_receipts']
        source_keys = [r['source_key'] for r in refs]
        need(all(digest(k) for k in source_keys) and len(set(source_keys)) == len(source_keys)
             and set(inventory(self.root/'sources')) == set(source_keys)
             and set(inventory(self.root/'commits')) == {k+'.json' for k in source_keys}, 'FROZEN_SOURCE_INVENTORY')
        self.samples, self.crop_positions, self._records = {}, {}, {}
        roles = {'train':[], 'val':[]}
        aliases, groups = set(), set()
        for ref in refs:
            key = ref['source_key']
            commit = safe_path(self.root,f'commits/{key}.json')
            need(commit.read_bytes()==canonical(dict(ref,plan_sha256=manifest['plan_sha256'])),
                 'SOURCE_COMMIT_PIN_MISMATCH')
            path = safe_path(self.root, f'sources/{key}/source.json')
            receipt = pinned(path,ref['sha256'])
            rid, oid, split, alias = (receipt[k] for k in ('image_id','oam_id','split','source_object_sha256'))
            fold, crop = receipt['validation_fold'], receipt['crop_xywh']
            need(type(rid) is int and rid not in self.samples and digest(alias) and alias not in aliases
                 and isinstance(oid,str) and oid and oid.casefold() not in groups
                 and type(fold) is int and 0 <= fold <= 4 and split == ('val' if fold==0 else 'train')
                 and receipt['source_hash_verification'] in AUTH & set(plan['allow_authenticity'])
                 and key == identifier([GENERATION,rid,oid,alias]), 'D6_SOURCE_IDENTITY')
            if real:
                source = planned[rid]
                need(type(source['image_id']) is int and all(receipt[k]==source[k] for k in (
                     'image_id','oam_id','split','validation_fold','source_object_sha256','crop_xywh',
                     'stac_sha256','source_hash_verification','historical_train_reuse'))
                     and receipt['input_sha256']==source['source_hashes']
                     and receipt['a1_output_shape']==source['output_shape']
                     and receipt['a1_crop_transform']==source['crop_transform'], 'D6_REAL_SOURCE_BINDING')
            validate_source(path.parent, receipt, dict(schema=RECEIPT,source_key=key,
                            patch_id=identifier([manifest['plan_sha256'],key,crop]),
                            plan_sha256=manifest['plan_sha256'],final_decision=ELIGIBLE))
            roles[split].append(rid)
            aliases.add(alias); groups.add(oid.casefold())
            self.samples[rid] = dict(image_id=rid,oam_id=oid,split=split,grid=receipt['full_grid'])
            self.crop_positions[rid] = [tuple(crop[:2])]
            self._records[rid] = (receipt,ref['sha256'],path)
        need(list(self.samples)==(list(planned) if real else plan['source_ids']) and roles==manifest['eligible_ids']
             and (shard or roles['train'] and roles['val']), 'FROZEN_ALLOWLIST')

    def patch(self, item, split):
        rid = item['image_id']
        need(type(rid) is int and rid in self._records, 'UNKNOWN_D6_SOURCE')
        receipt, pin, path = self._records[rid]
        crop = item['crop_xywh']
        need(receipt['split']==split and item['oam_id']==receipt['oam_id']
             and crop==receipt['crop_xywh'] and all(type(v) is int for v in crop), 'D6_PATCH_IDENTITY')
        pinned(safe_path(self.root,'manifest.json'),self.manifest_sha256)
        arrays = validate_source(path.parent,pinned(path,pin),receipt)
        return (torch.from_numpy(arrays['rgb.tif']).unsqueeze(0).to(torch.float32)/255,
                torch.from_numpy(arrays['canopy_mask.tif']).to(torch.long), arrays)
