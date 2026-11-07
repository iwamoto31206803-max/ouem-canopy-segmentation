"""R1a offline dataset contract. No acquisition, training or legacy mutation."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from pyproj import Geod

from m1_7_dataset import PRODUCTS, GENERATION, sha, require_hash

SCHEMA = 'ouem-r1-dataset-v1'


def canonical(manifest):
    core = copy.deepcopy(manifest)
    core.pop('dataset_id', None)
    core['samples'] = sorted(core['samples'], key=lambda s: s['image_id'])
    return json.dumps(core, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def identify(manifest):
    return 'ouem-r1-' + hashlib.sha256(canonical(manifest)).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def text(value):
    return isinstance(value, str) and bool(value.strip())


def safe_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative or any(
            p in ('', '.', '..') or p.endswith((' ', '.')) for p in relative.split('/')):
        raise ValueError('Unsafe relative path')
    # Reject symlinks AND Windows junction/reparse points before following them.
    for component in [root, *[root.joinpath(*relative.split('/')[:i])
                            for i in range(1, len(relative.split('/')) + 1)]]:
        for ancestor in [component, *component.parents]:
            if ancestor.exists() or ancestor.is_symlink():
                info = ancestor.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                    raise ValueError('Symlink/reparse point forbidden')
    path = root / relative
    path.resolve().relative_to(root.resolve())
    if not path.is_file():
        raise ValueError('Missing regular product file: ' + relative)
    return path


def contract(m):
    if not isinstance(m, dict): raise ValueError('Manifest must be an object')
    if m.get('schema') != SCHEMA or m.get('source_generation') != GENERATION:
        raise ValueError('Schema/source generation mismatch')
    samples = m.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('Positive sample count required')
    ids = []; paths = set()
    for s in samples:
        if not isinstance(s, dict): raise ValueError('Sample must be an object')
        for key in ('products', 'grid', 'source', 'rights', 'labels', 'category_review',
                    'validity_review', 'r1_review', 'independence_review', 'external_independence_review'):
            if key in s and not isinstance(s[key], dict): raise ValueError('Expected object: '+key)
        if type(s['image_id']) is not int or s['image_id'] < 0 or not text(s['oam_id']):
            raise ValueError('Invalid sample identity')
        ids.append(s['image_id'])
        if s['split'] not in ('train', 'val') or s['official_split'] != 'train':
            raise ValueError('Holdout/test or invalid split')
        fold = s['validation_fold']
        if type(fold) is not int or fold not in range(5) or s['split'] != ('val' if fold == 0 else 'train'):
            raise ValueError('Fold/split mismatch')
        if s['source_generation'] != GENERATION or s['threshold'] != .5:
            raise ValueError('Generation/threshold mismatch')
        if set(s['products']) != PRODUCTS:
            raise ValueError('Four products required')
        for name, product in s['products'].items():
            if not isinstance(product, dict): raise ValueError('Product must be an object')
            p = product['path']
            if not isinstance(p, str) or p != f"{s['split']}/{s['image_id']}/{name}" or p.casefold() in paths:
                raise ValueError('Unsafe/duplicate product path')
            paths.add(p.casefold())
            if not digest(product['sha256']):
                raise ValueError('Invalid product SHA256')
        grid = s['grid']
        if not all(isinstance(grid[k], list) for k in ('shape', 'transform', 'bounds')):
            raise ValueError('Grid vectors must be lists')
        if len(grid['shape']) != 2 or any(type(v) is not int or v <= 0 for v in grid['shape']):
            raise ValueError('Invalid shape')
        if len(grid['transform']) != 6 or len(grid['bounds']) != 4 or not all(
                type(v) in (int, float) and math.isfinite(v) for v in grid['transform'] + grid['bounds']):
            raise ValueError('Invalid affine/bounds')
        t = Affine(*grid['transform'])
        if not t.determinant or grid['bounds'][0] >= grid['bounds'][2] or grid['bounds'][1] >= grid['bounds'][3]:
            raise ValueError('Degenerate grid')
        rasterio.crs.CRS.from_user_input(grid['crs'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image_id')
    if m.get('dataset_id') != identify(m):
        raise ValueError('Dataset identity mismatch')


def products(root, sample, checks=None):
    """One sample only; hash again on read. Returns arrays plus coverage HOLD reasons."""
    arrays = {}; holds = []; grid = sample['grid']
    if checks is None: checks = []
    h, w = grid['shape']
    for name in sorted(sample['products']):
        product = sample['products'][name]
        entry = dict(product=name, status='FAIL', reasons=[])
        checks.append(entry)
        try:
            path = safe_path(Path(root), product['path']); require_hash(path, product['sha256'])
            with rasterio.open(path) as src:
                if (src.crs != rasterio.crs.CRS.from_user_input(grid['crs']) or
                        src.transform != Affine(*grid['transform']) or list(src.shape) != grid['shape'] or
                        not np.allclose(src.bounds, grid['bounds'], rtol=0, atol=1e-8)):
                    raise ValueError('CRS/grid/bounds mismatch: ' + name)
                if src.nodata is not None or not np.all(src.read_masks() == 255):
                    entry['reasons'].append('PARTIAL_OR_AMBIGUOUS_RASTER_VALIDITY:' + name)
                a = src.read()
            count, dtype = (3, np.uint8) if name == 'rgb.tif' else (1, np.uint8 if name == 'canopy_mask.tif' else np.float32)
            if a.shape != (count, h, w) or a.dtype != dtype:
                raise ValueError('Channels/dtype mismatch: ' + name)
            if name != 'rgb.tif' and (not np.isfinite(a).all() or not ((a >= 0) & (a <= 1)).all()):
                raise ValueError('Pixel range: ' + name)
            if name == 'valid_fraction.tif' and not np.all(a == 1):
                entry['reasons'].append('PARTIAL_VALID_FRACTION')
            arrays[name] = a
            entry['status'] = 'HOLD' if entry['reasons'] else 'PASS'
            holds.extend(entry['reasons'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            entry['reasons'].append(str(exc))
            raise
    if not np.array_equal(arrays['canopy_mask.tif'], (arrays['canopy_fraction.tif'] >= .5).astype(np.uint8)):
        raise ValueError('Mask/fraction threshold mismatch')
    return arrays, holds


def admission(s):
    """Completeness of R1b assertions, never a substitute for rights/visual review."""
    reasons = []
    source = s.get('source', {})
    if not all(text(source.get(k)) for k in ('dataset', 'revision', 'original_source_id')) or not all(
            digest(source.get(k)) for k in ('image_sha256', 'annotation_sha256')):
        reasons.append('SOURCE_PROVENANCE_NOT_VERIFIED')
    if source.get('dataset') != 'restor/tcd Zenodo' or source.get('revision') != GENERATION:
        reasons.append('SOURCE_GENERATION_EVIDENCE_CONFLICT')
    if not footprint(source.get('footprint_wgs84')):
        reasons.append('ORIGINAL_SOURCE_FOOTPRINT_NOT_VERIFIED')
    else:
        tile = transform_bounds(s['grid']['crs'], 4326, *s['grid']['bounds'], densify_pts=21)
        extent = source['footprint_wgs84']
        if (not footprint(list(tile)) or extent[0] > tile[0]+1e-8 or extent[1] > tile[1]+1e-8 or
                extent[2] < tile[2]-1e-8 or extent[3] < tile[3]-1e-8):
            reasons.append('SOURCE_FOOTPRINT_GRID_CONFLICT')
    rights = s.get('rights', {})
    for key in ('image_license', 'annotation_license', 'original_source_license'):
        if rights.get(key) != 'CC-BY-4.0': reasons.append('RIGHTS_MISSING_OR_CONFLICTING:' + key)
    if not all(text(rights.get(k)) for k in ('attribution', 'evidence_ref')) or not digest(rights.get('evidence_sha256')):
        reasons.append('RIGHTS_EVIDENCE_NOT_VERIFIED')
    labels = s.get('labels', {})
    if labels.get('policy') != 'upstream_nonzero_binary_with_review' or labels.get('categories') != [1, 2]:
        reasons.append('LABEL_POLICY_OR_CATEGORIES_NOT_VERIFIED')
    for name in ('category_review', 'validity_review', 'r1_review', 'independence_review', 'external_independence_review'):
        review = s.get(name, {})
        if (review.get('status') != 'PASS' or not text(review.get('evidence_ref')) or
                not digest(review.get('evidence_sha256')) or review.get('source_hashes') !=
                {k: source.get(k) for k in ('image_sha256', 'annotation_sha256')}):
            reasons.append(name.upper() + '_NOT_VERIFIED')
    external = s.get('external_independence_review', {})
    if (external.get('scope') != 'official_holdout_and_unlisted_neighbor_sources' or
            not digest(external.get('catalog_sha256'))):
        if 'EXTERNAL_INDEPENDENCE_REVIEW_NOT_VERIFIED' not in reasons:
            reasons.append('EXTERNAL_INDEPENDENCE_REVIEW_NOT_VERIFIED')
    return reasons


def footprint(value):
    return (isinstance(value, list) and len(value) == 4 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value) and
            -180 <= value[0] < value[2] <= 180 and -90 <= value[1] < value[3] <= 90)


def independence(samples):
    issues = {s['image_id']: [] for s in samples}
    geod = Geod(ellps='WGS84')
    for i, a in enumerate(samples):
        for b in samples[i+1:]:
            same_source = a.get('source', {}).get('original_source_id')
            image_hash = a.get('source', {}).get('image_sha256')
            reason = None
            if (a['oam_id'] == b['oam_id'] or
                    (text(same_source) and same_source == b.get('source', {}).get('original_source_id')) or
                    (digest(image_hash) and image_hash == b.get('source', {}).get('image_sha256'))):
                reason = 'DUPLICATE_SOURCE' if a['split'] == b['split'] else 'CROSS_SPLIT_SOURCE_LEAK'
            elif a['split'] != b['split']:
                extents = []
                for s in (a, b):
                    tile = transform_bounds(s['grid']['crs'], 4326, *s['grid']['bounds'], densify_pts=21)
                    original = s.get('source', {}).get('footprint_wgs84')
                    if footprint(original):
                        tile = [min(tile[0], original[0]), min(tile[1], original[1]),
                                max(tile[2], original[2]), max(tile[3], original[3])]
                    extents.append(tile)
                x, y = extents
                # Dateline/invalid envelopes cannot establish independence.
                if any(v[0] >= v[2] or v[1] >= v[3] or not all(math.isfinite(n) for n in v) for v in (x, y)):
                    reason = 'SPATIAL_INDEPENDENCE_NOT_VERIFIED'
                elif max(x[0], y[0]) < min(x[2], y[2]) and max(x[1], y[1]) < min(x[3], y[3]):
                    reason = 'CROSS_SPLIT_FOOTPRINT_OVERLAP'
                else:
                    centers = [((v[0]+v[2])/2, (v[1]+v[3])/2) for v in (x, y)]
                    radii = [max(geod.inv(*c, lon, lat)[2] for lon in (v[0], v[2]) for lat in (v[1], v[3])) for c, v in zip(centers, (x, y))]
                    if geod.inv(*centers[0], *centers[1])[2] - sum(radii) < 1000:
                        reason = 'NEAR_OPPOSITE_SOURCE_WITHIN_1KM'
            if reason:
                for s in (a, b): issues[s['image_id']].append(reason)
    return issues


def verify(root, m):
    report = dict(schema=SCHEMA, dataset_id=m.get('dataset_id') if isinstance(m, dict) else None, technical='FAIL', acceptance='FAIL', samples=[],
                  scope='Offline product verification; R1b review assertions checked, not independently adjudicated')
    try:
        contract(m); leaks = independence(m['samples'])
    except (ValueError, KeyError, TypeError, OverflowError, AttributeError) as exc:
        report['reasons'] = [str(exc)]; return report
    for s in m['samples']:
        item = dict(image_id=s['image_id'], split=s['split'], technical='PASS', acceptance='HOLD', technical_reasons=[], product_checks=[])
        try:
            item['acceptance_reasons'] = admission(s) + leaks[s['image_id']]
            _, item['technical_reasons'] = products(root, s, item['product_checks'])
            if item['technical_reasons']: item['technical'] = 'HOLD'
            item['acceptance'] = 'PASS' if item['technical'] == 'PASS' and not item['acceptance_reasons'] else 'HOLD'
            if any(r.startswith('CROSS_SPLIT') for r in leaks[s['image_id']]): item['acceptance'] = 'FAIL'
        except (ValueError, OSError, KeyError, TypeError) as exc:
            item.update(technical='FAIL', acceptance='FAIL', technical_reasons=[str(exc)])
            item.setdefault('acceptance_reasons', ['EVIDENCE_CHECK_FAILED'])
        report['samples'].append(item)
    for key in ('technical', 'acceptance'):
        values = [s[key] for s in report['samples']]
        report[key] = 'FAIL' if 'FAIL' in values else 'HOLD' if 'HOLD' in values else 'PASS'
    entries = [p for s in report['samples'] for p in s['product_checks']]
    report['product_counts'] = dict(expected=4*len(m['samples']), attempted=len(entries),
                                  not_attempted=4*len(m['samples'])-len(entries),
                                  **{status.lower(): sum(e['status'] == status for e in entries) for status in ('PASS','HOLD','FAIL')})
    report['samples_fully_technical_pass'] = sum(s['technical'] == 'PASS' for s in report['samples'])
    report['independence'] = dict(
        within_manifest=dict(status='FAIL' if any(r.startswith('CROSS_SPLIT') for values in leaks.values() for r in values)
                             else 'HOLD' if any(leaks.values()) or any(not text(s.get('source', {}).get('original_source_id')) or not footprint(s.get('source', {}).get('footprint_wgs84')) for s in m['samples']) else 'PASS',
                             scope='Declared identity/envelope checks within this manifest only; no external catalog scan'),
        external_sources=dict(status='NOT_VERIFIED',
                              review_assertions_complete=all('EXTERNAL_INDEPENDENCE_REVIEW_NOT_VERIFIED' not in s['acceptance_reasons'] for s in report['samples']),
                              scope='Official holdout and unlisted neighboring/aliased sources are not inspected by this offline validator; R1b must supply source-bound catalog review evidence.'))
    return report


_PREFLIGHT_SEAL = object()


@dataclass(frozen=True)
class _VerifiedDataset:
    root: Path
    manifest_bytes: bytes
    seal: object


def preflight(root, m):
    """Only a complete technical + acceptance PASS issues a loading context."""
    snapshot = copy.deepcopy(m)
    report = verify(root, snapshot)
    if report['technical'] != 'PASS' or report['acceptance'] != 'PASS':
        raise ValueError('Dataset preflight HOLD/FAIL; inspect verify() report')
    return _VerifiedDataset(Path(root).resolve(), json.dumps(snapshot, allow_nan=False).encode('utf-8'), _PREFLIGHT_SEAL)


def load_sample(verified, image_id, expected_split):
    """Requires an in-process full preflight; selected bytes are checked again."""
    if not isinstance(verified, _VerifiedDataset) or verified.seal is not _PREFLIGHT_SEAL:
        raise ValueError('Successful dataset preflight context required')
    m = json.loads(verified.manifest_bytes)
    sample = next((s for s in m['samples'] if s['image_id'] == image_id), None)
    if sample is None or expected_split not in ('train', 'val') or sample['split'] != expected_split:
        raise ValueError('Sample/split mismatch')
    arrays, holds = products(verified.root, sample)
    if holds: raise ValueError('Validity HOLD')
    return arrays


def from_g0(old):
    """Read-only compatibility projection, deliberately no R1 acceptance inheritance."""
    from m1_7_dataset import check_manifest
    check_manifest(old)
    m = dict(schema=SCHEMA, source_generation=GENERATION, legacy_dataset_id=old['dataset_id'], samples=[])
    for s in old['samples']:
        stem = s['oam_id'] + '_' + str(s['image_id'])
        item = {k: copy.deepcopy(s[k]) for k in ('image_id', 'oam_id', 'split', 'official_split', 'validation_fold', 'source_generation', 'threshold')}
        item.update(grid={k: s['output_'+k] for k in ('crs', 'transform', 'bounds', 'shape')},
                    products={name: dict(path=f"{s['split']}/{s['image_id']}/{name}", sha256=h) for name, h in s['output_hashes'].items()},
                    source=dict(dataset='restor/tcd Zenodo', revision=GENERATION, image_sha256=s['source_hashes'][stem+'.tif'], annotation_sha256=s['source_hashes'][stem+'.png']),
                    legacy_evidence=s['acceptance_evidence'], legacy_rights=s['rights'],
                    labels=dict(policy=s['semantic_policy'], categories=s['coco_categories'], per_segment_category_status=s['category_level_per_id_validation']))
        m['samples'].append(item)
    m['dataset_id'] = identify(m)
    return m


def read_manifest(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--legacy-g0', action='store_true')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args(argv)
    # Exclusive creation protects prior reports even when validation fails.
    with args.report.open('x', encoding='utf-8') as stream:
        try:
            require_hash(args.manifest, args.manifest_sha256)
            m = read_manifest(args.manifest)
            if args.legacy_g0: m = from_g0(m)
            report = verify(args.root, m)
            report.update(input_manifest_sha256=args.manifest_sha256,
                          legacy_g0_projection=args.legacy_g0)
            if args.legacy_g0: report['legacy_dataset_id'] = m['legacy_dataset_id']
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            report = dict(technical='FAIL', acceptance='FAIL', reasons=[str(exc)])
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('technical', 'acceptance')}))
    return 1 if 'FAIL' in (report['technical'], report['acceptance']) else 2 if report['acceptance'] == 'HOLD' else 0


if __name__ == '__main__':
    raise SystemExit(main())
