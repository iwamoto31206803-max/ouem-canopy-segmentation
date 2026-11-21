"""Offline, research-only OAM-TCD eligibility and crop recipes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

from m1_2d_geometry_audit import require_hash, sha
from m1_2d_prepare_pairs import output_grid
from m1_5b_pilot import rights, separation
from m1_5c_zenodo_audit import coco_annotation_summary, json_lines, partition
from m1_5d_zenodo_prepare import GENERATION, semantic_binary
from r1b_candidate_plan import identity
from r1b_qa_runner import stac_structure, strict_json
from research_patches import POLICY as CROP_POLICY, choose_window, crop_arrays


POLICY = 'd6a-research-policy-v1'
ELIGIBLE = 'AUTO_ELIGIBLE_RESEARCH'
HOLD = 'AUTO_HOLD'
EXCLUDE = 'AUTO_EXCLUDE'
FAIL = 'TECHNICAL_FAIL'
FILES = dict(train='archives/train_meta.json', test='archives/test_meta.json',
             coco='archives/dataset/holdout/train.json')
LABEL_POLICY = [
    'UPSTREAM_OAM_TCD_LABELS_ACCEPTED_AS_IS_FOR_EXPLORATORY_RESEARCH',
    'ANNOTATION_SEMANTIC_ACCURACY_NOT_INDEPENDENTLY_VERIFIED',
]
STAC_PREFIX = 'https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'
MAX_PIXELS = 2048**2


class Decision(Exception):
    def __init__(self, status, reason, observed=None):
        self.result = dict(status=status, reason=reason, observed=observed)


def check(condition, status, reason, observed=None):
    if not condition:
        raise Decision(status, reason, observed)


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def local_file(root, name):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe local input path')
    return path


def load_catalog(paths):
    rows = {}
    groups = defaultdict(list)
    object_folds = defaultdict(set)
    for official in ('train', 'test'):
        for row in json_lines(paths[official]):
            rid, group, fold = (row[k] for k in ('image_id', 'oam_id', 'validation_fold'))
            if (type(rid) is not int or rid < 0 or rid in rows or type(fold) is not int
                    or not isinstance(group, str) or not re.fullmatch('[A-Za-z0-9-]+', group)):
                raise ValueError('Invalid or duplicate catalog identity')
            role = partition(row, official)
            box = row.get('bounds')
            geographic = None
            if (isinstance(box, list) and len(box) == 4
                    and all(type(v) in (int, float) and np.isfinite(v) for v in box)
                    and box[0] < box[2] and box[1] < box[3] and row.get('crs')):
                try:
                    candidate = transform_bounds(row['crs'], 4326, *box)
                except (ValueError, rasterio.errors.CRSError):
                    candidate = (0, 0, 0, 0)
                if (-180 <= candidate[0] < candidate[2] <= 180
                        and -90 <= candidate[1] < candidate[3] <= 90):
                    geographic = candidate
            row = dict(row, official_split=official, split=role, wgs84_bounds=geographic)
            rows[rid] = row
            groups[group].append(row)
            object_folds[identity(row)].add((role, fold))
    envelopes = {}
    for group, members in groups.items():
        parts = {r['split'] for r in members}
        bounds = [r['wgs84_bounds'] for r in members]
        envelope = center = None
        if all(bounds):
            envelope = [min(b[0] for b in bounds), min(b[1] for b in bounds),
                        max(b[2] for b in bounds), max(b[3] for b in bounds)]
            center = [(envelope[0]+envelope[2])/2, (envelope[1]+envelope[3])/2]
        envelopes[group] = dict(members[0], wgs84_bounds=envelope, center=center, roles=parts,
                                folds={r['validation_fold'] for r in members},
                                source_objects={identity(r) for r in members})
    return rows, envelopes, object_folds


def source_preflight(row, candidate, groups, object_folds, duplicates, evidence):
    check(row['official_split'] == 'train', EXCLUDE, 'OFFICIAL_TEST_OR_HOLDOUT')
    check(all(row[k] == candidate[k] for k in ('oam_id', 'validation_fold')),
          HOLD, 'CANDIDATE_IDENTITY_CONFLICT')
    check(row.get('license') == 'CC-BY 4.0', HOLD, 'CC_BY_4_0_DECLARATION_REQUIRED')
    group = groups[row['oam_id']]
    source = identity(row)
    check(source is not None and group['source_objects'] == {source}, HOLD, 'ORIGINAL_OBJECT_UNRESOLVED')
    check(len(group['roles']) == 1 and len(group['folds']) == 1
          and object_folds[source] == {(row['split'], row['validation_fold'])},
          HOLD, 'CROSS_SPLIT_SOURCE_IDENTITY')
    check(row['image_id'] not in duplicates, EXCLUDE, 'DUPLICATE_POOL_SOURCE_GROUP')
    opposite = [g for g in groups.values() if g['roles'] - {row['split']}]
    check(bool(opposite) and group['wgs84_bounds'] is not None
          and all(g['wgs84_bounds'] is not None and len(g['roles']) == 1 for g in opposite),
          HOLD, 'OPPOSITE_GEOGRAPHY_UNRESOLVED')
    try:
        guard = separation(dict(row, wgs84_bounds=group['wgs84_bounds'], center=group['center']), opposite)
    except ValueError as exc:
        reason = str(exc)
        if reason not in {'HOLD_CROSS_SPLIT_FOOTPRINT', 'HOLD_NEAR_OPPOSITE_SOURCE_WITHIN_1KM'}:
            reason = 'CATALOG_SPATIAL_CONFLICT'
        raise Decision(HOLD, reason, {'proximity_guard_m': 1000}) from None
    pin = candidate.get('stac_sha256')
    check(digest(pin), HOLD, 'STAC_PIN_MISSING')
    path = local_file(evidence, row['oam_id'] + '-stac.json')
    check(path.is_file(), HOLD, 'STAC_EVIDENCE_MISSING')
    check(sha(path) == pin, FAIL, 'STAC_HASH_MISMATCH')
    try:
        stac = stac_structure(path.read_bytes(), row)
        _, declaration = rights(dict(row, source_evidence=dict(
            snapshot_sha256=pin, url=STAC_PREFIX + row['oam_id'])), evidence)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError):
        raise Decision(HOLD, 'STAC_DECLARATION_CONFLICT') from None
    b, tile = stac['bbox'], row['wgs84_bounds']
    check(tile is not None and b[0] <= tile[0] < tile[2] <= b[2] and b[1] <= tile[1] < tile[3] <= b[3],
          HOLD, 'TILE_OUTSIDE_STAC_DECLARED_BBOX')
    return dict(source_object_sha256=source, stac_sha256=declaration['stac_sha256'],
                license_declarations={k: declaration[k] for k in (
                    'row_license', 'image_license', 'annotation_license')},
                catalog_separation=guard)


def decode(row, paths, categories):
    with paths['annotation'].open('rb') as stream:
        header = stream.read(29)
    with Image.open(paths['rgb']) as image, Image.open(paths['annotation']) as ann:
        check(image.format == 'TIFF' and ann.format == 'PNG', FAIL, 'TIFF_PNG_FORMAT')
        check('A' not in image.getbands() and 'A' not in ann.getbands()
              and ann.info.get('transparency') is None, HOLD, 'DECLARED_ALPHA_OR_TRANSPARENCY')
        check(image.mode == ann.mode == 'RGB' and header[24:26] == b'\x08\x02',
              FAIL, 'RGB_UINT8_ENCODING', {'tiff_mode': image.mode, 'png_mode': ann.mode})
        dims = (row.get('width'), row.get('height'))
        check(all(type(v) is int and v > 0 for v in dims) and image.size == ann.size == dims,
              FAIL, 'IMAGE_MASK_METADATA_DIMENSIONS', {'image': list(image.size), 'mask': list(ann.size)})
        check(image.width * image.height <= MAX_PIXELS, HOLD, 'INPUT_PIXEL_LIMIT', image.width * image.height)
        check(all(p.getexif().get(274, 1) == 1 for p in (image, ann)), FAIL, 'IMAGE_ORIENTATION')
        rgb, panoptic = np.array(image), np.array(ann)
    try:
        teacher = semantic_binary(rgb, panoptic, categories)
    except ValueError:
        raise Decision(FAIL, 'LABEL_ENCODING_OR_CATEGORY_CONFLICT') from None
    # The primary TIFF hash does not bind an external mask or auxiliary grid.
    check(not any(Path(str(paths['rgb']) + ext).exists() for ext in ('.msk', '.aux.xml', '.ovr')),
          HOLD, 'EXTERNAL_RASTER_SIDECAR')
    with rasterio.open(paths['rgb']) as src:
        check(len(src.files) == 1, HOLD, 'EXTERNAL_RASTER_SIDECAR')
        t = src.transform
        check(str(src.crs) == row['crs'] == 'EPSG:3395' and src.count == 3
              and src.dtypes == ('uint8',)*3 and src.shape == rgb.shape[:2]
              and not t.b and not t.d and t.a > 0 and t.e < 0
              and np.isfinite(list(t)).all() and np.allclose(src.bounds, row['bounds'], rtol=0, atol=1e-8),
              FAIL, 'GRID_CRS_OR_METADATA_BOUNDS')
        check(src.nodata is None and not any(c.name == 'alpha' for c in src.colorinterp)
              and np.all(src.read_masks() == 255), HOLD, 'DECLARED_NODATA_ALPHA_OR_INVALID_MASK')
        check(tuple(c.name for c in src.colorinterp) == ('red', 'green', 'blue')
              and np.array_equal(src.read().transpose(1, 2, 0), rgb), FAIL, 'RASTER_RGB_CHANNEL_MISMATCH')
        shape, grid, gsd = output_grid(src.crs, t, src.width, src.height)
    return rgb, teacher, shape, grid, gsd


def inspect(candidate, row, groups, object_folds, duplicates, categories, root, evidence, seed):
    result = dict(image_id=candidate['image_id'], oam_id=candidate['oam_id'],
                  validation_fold=candidate['validation_fold'], window_decision=None)
    try:
        check(row is not None, EXCLUDE, 'OUTSIDE_PINNED_CATALOG')
        result.update(split=row['split'], official_split=row['official_split'])
        result.update(source_preflight(row, candidate, groups, object_folds, duplicates, evidence))
        stem = f"{row['oam_id']}_{row['image_id']}"
        paths = {key: local_file(root, f'extracted/dataset/holdout/{folder}/{stem}{suffix}')
                 for key, folder, suffix in [('rgb', 'images', '.tif'), ('annotation', 'masks', '.png')]}
        check(all(p.is_file() for p in paths.values()), HOLD, 'LOCAL_SOURCE_FILE_MISSING')
        hashes = {k: sha(p) for k, p in paths.items()}
        pins = candidate.get('source_hashes')
        result.update(source_hashes=hashes, source_hash_basis='EXTERNAL_PINS' if pins else 'LOCAL_OBSERVED')
        check(pins is None or hashes == pins, FAIL, 'SOURCE_HASH_MISMATCH')
        rgb, teacher, shape, grid, gsd = decode(row, paths, categories)
        check(hashes == {k: sha(p) for k, p in paths.items()}, FAIL, 'SOURCE_BYTES_CHANGED')
    except Decision as exc:
        result['source_decision'] = exc.result
        return result
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        result['source_decision'] = dict(status=FAIL, reason='MALFORMED_SOURCE_OR_READ_ERROR', observed=None)
        return result
    result['source_decision'] = dict(status=ELIGIBLE, reason='SOURCE_TECHNICAL_AND_DOCUMENTARY_CHECKS', observed=None)
    result.update(source_canopy_fraction=float(teacher.mean()), output_shape=list(shape),
                  output_ground_gsd=gsd, crs=row['crs'], coco_categories=categories)
    try:
        check(all(a <= b for a, b in zip(shape, teacher.shape)), HOLD, 'UPSCALE_REQUIRED', list(shape))
        try:
            selection = choose_window(rgb, teacher, shape, row['image_id'], seed=seed)
        except ValueError as exc:
            if str(exc) != 'No direct clean window; padding/upscale forbidden':
                raise
            raise Decision(HOLD, 'NO_CLEAN_224_WINDOW', {'output_shape': list(shape), 'placements': 0}) from None
        check(selection['selected_black_support_cells'] == 0, FAIL, 'CROP_INTERSECTS_BLACK_SUPPORT')
        crop = selection['crop_xywh']
        arrays, transform = crop_arrays(rgb, teacher, shape, grid, crop)
        fraction = arrays['canopy_fraction.tif']
        mask = arrays['canopy_mask.tif']
        check(fraction.shape == mask.shape == (1, 224, 224) and np.isfinite(fraction).all()
              and np.array_equal(mask, fraction >= .5), FAIL, 'CROP_PRODUCT_INTEGRITY')
        result.update(crop_xywh=crop, crop_transform=list(transform)[:6], placement_count=selection['placement_count'],
                      border_black_pixels=selection['black']['border_connected_pixels'],
                      selected_black_support_cells=selection['selected_black_support_cells'],
                      selected_white_support_cells=selection['selected_white_support_cells'],
                      crop_canopy_fraction=float(fraction.mean()), crop_binary_fraction=float(mask.mean()))
        result['window_decision'] = dict(status=ELIGIBLE, reason='CLEAN_224_WINDOW', observed=0)
    except Decision as exc:
        result['window_decision'] = exc.result
    except (ValueError, OSError, RuntimeError):
        result['window_decision'] = dict(status=FAIL, reason='CROP_PROCESSING_ERROR', observed=None)
    return result


def assess(root, evidence, contract, contract_sha256):
    root, evidence = root.resolve(strict=True), evidence.resolve(strict=True)
    require_hash(contract, contract_sha256)
    policy = strict_json(contract.read_bytes())
    if (set(policy) != {'schema', 'source_generation', 'seed', 'catalog_sha256', 'candidates'}
            or policy['schema'] != POLICY or policy['source_generation'] != GENERATION
            or type(policy['seed']) is not int or not 0 <= policy['seed'] < 2**32
            or set(policy['catalog_sha256']) != FILES.keys()
            or not isinstance(policy['candidates'], list) or not policy['candidates']):
        raise ValueError('Invalid D6 policy contract')
    paths = {key: local_file(root, filename) for key, filename in FILES.items()}
    for key, path in paths.items():
        if not digest(policy['catalog_sha256'][key]):
            raise ValueError('Missing catalog pin')
        require_hash(path, policy['catalog_sha256'][key])
    rows, groups, object_folds = load_catalog(paths)
    candidates = {}
    for c in policy['candidates']:
        required = {'image_id', 'oam_id', 'validation_fold'}
        if (not isinstance(c, dict) or not required <= c.keys() <= required | {'source_hashes', 'stac_sha256'}
                or type(c['image_id']) is not int or c['image_id'] < 0 or c['image_id'] in candidates
                or not isinstance(c['oam_id'], str) or not re.fullmatch('[A-Za-z0-9-]+', c['oam_id'])
                or type(c['validation_fold']) is not int or c['validation_fold'] not in range(-1, 5)
                or ('source_hashes' in c and (not isinstance(c['source_hashes'], dict)
                    or set(c['source_hashes']) != {'rgb', 'annotation'}
                    or not all(digest(h) for h in c['source_hashes'].values())))):
            raise ValueError('Malformed or duplicate candidate')
        candidates[c['image_id']] = c
    targets = [rows[i] for i in candidates if i in rows and rows[i]['official_split'] == 'train']
    oams = Counter(r['oam_id'] for r in targets)
    objects = Counter(identity(r) for r in targets if identity(r))
    duplicates = {r['image_id'] for r in targets if oams[r['oam_id']] > 1 or objects[identity(r)] > 1}
    annotations = coco_annotation_summary(paths['coco'], [r['image_id'] for r in targets])
    results = [inspect(c, rows.get(i), groups, object_folds, duplicates, annotations.get(i, {}).get('categories', []),
                       root, evidence, policy['seed']) for i, c in sorted(candidates.items())]
    require_hash(contract, contract_sha256)
    for key, path in paths.items():
        require_hash(path, policy['catalog_sha256'][key])
    return dict(policy=POLICY, purpose='EXPLORATORY_RESEARCH_ONLY', formal_r1='UNCHANGED',
                source_generation=GENERATION, contract_sha256=contract_sha256,
                seed=policy['seed'], crop_policy=CROP_POLICY, labels=LABEL_POLICY,
                category_crosswalk='NOT_VERIFIED', photographed_coverage='NOT_VERIFIED',
                catalog_sha256=policy['catalog_sha256'], catalog_rows=len(rows),
                catalog_scope='Pinned official Train and Test metadata; Test pixels excluded',
                external_independence='UNLISTED_SOURCES_NOT_VERIFIED',
                counts=dict(Counter((r['window_decision'] or r['source_decision'])['status'] for r in results)),
                results=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'evidence', 'contract', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        protected = (args.root.resolve(), args.evidence.resolve(), args.contract.resolve(),
                     Path(__file__).resolve().parents[1])
        if (args.output.is_symlink() or output.exists() or output.suffix != '.json'
                or not output.parent.is_dir() or any(output.is_relative_to(p) for p in protected)
                or not digest(args.contract_sha256)):
            raise ValueError('Fresh external JSON output and contract pin required')
        report = assess(args.root, args.evidence, args.contract, args.contract_sha256)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    except (Decision, ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'D6 preflight or report failed: {type(exc).__name__}\n')
    return int(FAIL in report['counts'])


if __name__ == '__main__':
    raise SystemExit(main())
