"""Phase C2 research-only reader for the pinned five-source C1 pilot."""
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
import torch

import r1_dataset as r1
from research_patches import TARGETS, POLICY, GENERATION, read_candidates


PRODUCTS = {'rgb.tif': ('uint8', 3), 'canopy_fraction.tif': ('float32', 1),
            'canopy_mask.tif': ('uint8', 1)}
DECISIONS = {'CLEAR_FOR_EXPLORATION', 'REVIEW_REQUIRED', 'EXCLUDE_FROM_FIRST_TRAINING'}


def https(value):
    if not r1.text(value):
        return False
    url = urlsplit(value)
    return url.scheme == 'https' and bool(url.hostname) and not url.username and not url.password


def affine(values):
    if (not isinstance(values, list) or len(values) != 6
            or any(type(v) not in (int, float) for v in values)
            or not np.isfinite(values).all()):
        raise ValueError('Invalid affine')
    result = Affine(*values)
    if result.b or result.d or result.a <= 0 or result.e >= 0:
        raise ValueError('North-up grid required')
    return result


class ResearchData:
    """Explicit review subset, independent of R1a preflight and acceptance.

    The caller pins both the immutable C1 plan and the research review. A clear
    decision allows exploratory input only. Product bytes are rechecked at use.
    """
    def __init__(self, root, plan_sha256, review, review_sha256, *, candidates=None, candidates_sha256=None):
        self.root = Path(root).resolve()
        self.plan_path = r1.safe_path(self.root, 'crop_plan.json')
        self.review_path = Path(review)
        r1.require_hash(self.plan_path, plan_sha256)
        r1.require_hash(self.review_path, review_sha256)
        plan = r1.read_manifest(self.plan_path)
        reviews = r1.read_manifest(self.review_path)
        self.expanded = plan.get('schema') == 'ouem-exploratory-crops-v2'
        targets = TARGETS
        candidate_records = {}
        if self.expanded:
            candidate_value, candidate_records = read_candidates(candidates, candidates_sha256)
            ids = plan['target_ids']
            if (not ids or any(type(i) is not int or i not in candidate_records for i in ids)
                    or len(set(ids)) != len(ids) or plan['candidate_manifest_sha256'] != candidates_sha256
                    or plan['diagnosis_sha256'] != candidate_value['phase_b_sha256']
                    or plan['metadata_sha256'] != candidate_value['metadata_sha256']):
                raise ValueError('Candidate/plan target mismatch')
            targets = {rid: candidate_records[rid]['split'] for rid in ids}
        self.targets = dict(targets)
        if (plan.get('schema') != ('ouem-exploratory-crops-v2' if self.expanded else 'ouem-exploratory-crops-v1')
                or plan.get('purpose') != ('PHASE_C3_RESEARCH_ONLY' if self.expanded else 'PHASE_C1_RESEARCH_ONLY')
                or plan.get('status') != 'EXPLORATORY_PILOT_PRODUCTS_COMPLETE'
                or plan.get('source_generation') != GENERATION or plan.get('policy') != POLICY
                or plan.get('r1_acceptance') != 'UNCHANGED_HOLD'):
            raise ValueError('Research schema/source/status mismatch')
        if (reviews.get('schema') != ('ouem-research-patch-review-v2' if self.expanded else 'ouem-research-patch-review-v1')
                or reviews.get('crop_plan_sha256') != plan_sha256
                or reviews.get('formal_r1') != 'UNCHANGED_HOLD'
                or not r1.text(reviews.get('reviewer'))):
            raise ValueError('Research review binding required')
        self._records = {}
        for sample in plan['samples']:
            rid = sample['image_id']
            key = self._key(sample)
            if type(rid) is not int or rid not in targets or key in self._records:
                raise ValueError('Unexpected/duplicate image ID')
            self._records[key] = sample
        decisions = {}
        for entry in reviews['samples']:
            rid = entry['image_id']
            key = self._key(entry)
            if (type(rid) is not int or rid not in targets or key in decisions
                    or entry.get('decision') not in DECISIONS or not r1.text(entry.get('reason'))):
                raise ValueError('Explicit per-image review decision required')
            decisions[key] = entry['decision']
        if (set(s['image_id'] for s in self._records.values()) != set(targets)
                or set(decisions) != set(self._records)):
            raise ValueError('Complete source/window plan/review required')
        sources = {}
        by_source = {}
        for sample in self._records.values():
            rid = sample['image_id']
            fold = sample['validation_fold']
            if (sample['split'] != targets[rid] or type(fold) is not int or fold not in range(5)
                    or sample['split'] != ('val' if fold == 0 else 'train')
                    or not r1.text(sample['oam_id']) or sources.get(sample['oam_id'], rid) != rid):
                raise ValueError('Source/fold/split mismatch')
            sources[sample['oam_id']] = rid
            first = by_source.setdefault(rid, sample)
            if self.expanded and any(sample[k] != first[k] for k in ('oam_id','split','validation_fold','source_hashes',
                    'source_metadata','source_grid','full_grid','rights')):
                raise ValueError('Conflicting windows from same source')
            if self.expanded:
                expected = candidate_records[rid]
                if (any(sample[k] != expected[k] for k in ('oam_id','split','validation_fold','source_hashes','source_metadata'))
                        or sample['full_grid']['shape'] != expected['output_shape']
                        or sample['placement_count'] != expected['placement_count']):
                    raise ValueError('Window/candidate provenance mismatch')
            source_reference = https(sample['source_metadata'])
            if self.expanded and r1.text(sample['source_metadata']):
                # Historical official metadata uses HTTP identifiers. No fetch
                # occurs here; retain the exact candidate-pinned object path.
                url = urlsplit(sample['source_metadata'])
                source_reference = source_reference or (url.scheme == 'http'
                    and url.hostname == 'oin-hotosm.s3.amazonaws.com'
                    and url.path.endswith('_meta.json') and not url.username and not url.password)
            if (set(sample['source_hashes']) != {'rgb', 'annotation'}
                    or not all(r1.digest(v) for v in sample['source_hashes'].values())
                    or not source_reference):
                raise ValueError('Source hashes/metadata required')
            rights = sample['rights']
            for key in ('row_license', 'image_license', 'annotation_license'):
                if rights.get(key, '').lower() not in ('cc-by-4.0', 'cc-by 4.0', 'cc by 4.0'):
                    raise ValueError('Explicit CC BY 4.0 required')
            if (not all(r1.text(rights.get(k)) for k in ('producer', 'attribution'))
                    or not r1.digest(rights.get('stac_sha256'))
                    or rights.get('stac_url') != 'https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+sample['oam_id']
                    or not https(rights.get('annotation_terms'))):
                raise ValueError('Source-bound rights/attribution required')
            self._read(sample)
        self._allowed = frozenset(key for key in self._records if decisions[key] == 'CLEAR_FOR_EXPLORATION')
        allowed_ids = {self._records[key]['image_id'] for key in self._allowed}
        self.samples = {rid: dict(image_id=rid, oam_id=by_source[rid]['oam_id'], split=targets[rid],
                                 grid={'shape': by_source[rid]['full_grid']['shape'] if self.expanded else [224,224]})
                        for rid in sorted(allowed_ids)}
        if self.expanded:
            self.crop_positions = {rid: sorted({tuple(self._records[key]['crop_xywh'][:2])
                                    for key in self._allowed if self._records[key]['image_id'] == rid})
                                   for rid in self.samples}
            if any(len(self.crop_positions[rid]) != 1 for rid in self.samples if targets[rid] == 'val'):
                raise ValueError('Exactly one fixed reviewed Val window required per source')
        self.identity = dict(dataset_id='ouem-research-'+r1.sha(self.plan_path),
                             crop_plan_sha256=plan_sha256, research_review_sha256=review_sha256,
                             source_generation=GENERATION, purpose='EXPLORATORY_RESEARCH_ONLY',
                             formal_r1='UNCHANGED_HOLD')
        if self.expanded:
            self.candidates_path = Path(candidates)
            self.identity['candidate_manifest_sha256'] = candidates_sha256

    def _key(self, sample):
        if not self.expanded:
            return sample['image_id']
        crop = sample['crop_xywh']
        if len(crop) != 4 or any(type(v) is not int for v in crop):
            raise ValueError('Integer window coordinates required')
        return (sample['image_id'], *crop)

    def _read(self, sample):
        x, y, w, h = sample['crop_xywh']
        shape = sample['full_grid']['shape']
        if (any(type(v) is not int for v in (x, y, w, h, *shape)) or len(shape) != 2
                or min(x, y) < 0 or (w, h) != (224, 224)
                or x+w > shape[1] or y+h > shape[0]):
            raise ValueError('Invalid fixed 224 crop')
        grid = sample['patch_grid']
        transform = affine(grid['affine'])
        full = affine(sample['full_grid']['affine'])
        if (grid['shape'] != [224, 224] or any(type(v) is not int for v in grid['shape'])
                or transform != full*Affine.translation(x, y)
                or not np.allclose(grid['bounds'], array_bounds(224, 224, transform), rtol=0, atol=1e-8)
                or set(sample['products']) != set(PRODUCTS)
                or sample['selected_black_support_cells'] != 0):
            raise ValueError('Crop/grid/black support mismatch')
        arrays = {}
        for name, (dtype, count) in PRODUCTS.items():
            product = sample['products'][name]
            relative = f"{sample['split']}/{sample['image_id']}"
            if self.expanded:
                relative += f'/crop-{x}-{y}'
            if product['path'] != f'{relative}/{name}':
                raise ValueError('Unexpected product path')
            path = r1.safe_path(self.root, product['path'])
            r1.require_hash(path, product['sha256'])
            with rasterio.open(path) as src:
                if (src.shape != (224, 224) or src.count != count or src.dtypes != (dtype,)*count
                        or src.crs is None or src.crs != rasterio.crs.CRS.from_user_input(grid['crs'])
                        or src.transform != transform or src.nodata is not None
                        or not np.all(src.dataset_mask() == 255)
                        or not np.allclose(src.bounds, grid['bounds'], rtol=0, atol=1e-8)):
                    raise ValueError('Product format/grid mismatch')
                arrays[name] = src.read()
            r1.require_hash(path, product['sha256'])
        fraction = arrays['canopy_fraction.tif']
        mask = arrays['canopy_mask.tif']
        if (not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1))
                or not np.array_equal(mask, (fraction >= .5).astype(np.uint8))):
            raise ValueError('Fraction range/saved threshold mismatch')
        return arrays

    def patch(self, item, split):
        r1.require_hash(self.plan_path, self.identity['crop_plan_sha256'])
        r1.require_hash(self.review_path, self.identity['research_review_sha256'])
        if self.expanded:
            r1.require_hash(self.candidates_path, self.identity['candidate_manifest_sha256'])
        rid = item['image_id']
        key = self._key(item)
        if type(rid) is not int or key not in self._allowed or split != self.targets[rid]:
            raise ValueError('Unreviewed source or split mismatch')
        sample = self._records[key]
        expected_crop = sample['crop_xywh'] if self.expanded else [0,0,224,224]
        if (item['oam_id'] != sample['oam_id'] or item['crop_xywh'] != expected_crop
                or any(type(v) is not int for v in item['crop_xywh'])):
            raise ValueError('Fixed research patch identity/crop required')
        arrays = self._read(sample)
        images = torch.from_numpy(arrays['rgb.tif']).unsqueeze(0).to(torch.float32)/255
        target = torch.from_numpy(arrays['canopy_mask.tif']).to(torch.long)
        return images, target, arrays
