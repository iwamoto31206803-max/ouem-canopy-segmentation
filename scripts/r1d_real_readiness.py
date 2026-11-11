"""Read-only fixed-four evidence preflight. No image execution entry point."""
import argparse
import json
from pathlib import Path
import re

from r1d_bounded_prepare import (GENERATION, LIMITS, TARGETS, Stop,
                                 digest, ordinary, relative_path)


CATALOG_SHA = '0bb9e9501792669d5365a25d11a6e354c410cc72b5cdf4cba73ac784ddfe37a0'
OAMS = ['5cb64e020a3a5c0005ff12ff', '5e25ecee36067e0005c2752e',
        '5dde8d370140b80006d2e547', '5b925da45d406700087d1514']
SHAPES = [390, 391, 394, 337]
FOLDS = [2, 3, 0, 0]


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def checked_snapshot(path, expected, limit=65536):
    """Hash and parse the same bounded snapshot, retaining duplicate-key rejection."""
    ordinary(path)
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'JSON read ceiling exceeded')
    require(digest(data) == expected, 'Input file SHA mismatch')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise Stop('Nonfinite JSON')
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique, parse_constant=nonfinite)


def read_receipt(root, reference):
    """Only bounded JSON receipts are opened. Embedded paths are never followed."""
    name = relative_path(reference['path'])
    require(name.endswith('.json'), 'JSON receipt required')
    root = Path(root).absolute()
    path = root / name
    require(root.resolve() in path.resolve().parents, 'Receipt outside evidence root')
    return checked_snapshot(path, reference['sha256'], 4 * 1024**2)


def check_receipts(package, catalog, root):
    receipts = {key: read_receipt(root, ref)
                for key, ref in package['evidence'].items()}
    archive = receipts['archive']
    require(archive['complete'] is True and archive['source_generation'] == GENERATION,
            'Archive receipt incomplete')
    members = []
    for item in archive['archives']:
        require(item['checksum_match'] is True and item['gzip_stream_integrity'] is True
                and item['normalized_paths_unique'] is True, 'Archive identity unresolved')
        members.extend(item['target_members'])
    require(len(members) == 8 and len({m['member'] for m in members}) == 8,
            'Eight unique target members required')
    require(receipts['independence']['hard_stop'] is False, 'Independence HARD STOP')
    require(receipts['independence']['catalog_sha256'] ==
            {item['name']: item['sha256'] for item in catalog['metadata']},
            'Catalog receipt mismatch')
    for kind, key in [('rights', 'images'), ('independence', 'results'), ('coverage', 'images')]:
        rows = receipts[kind][key]
        require(len(rows) == 4 and {r['image_id'] for r in rows} == set(TARGETS),
                'Receipt targets mismatch')
        for sample in package['samples']:
            row = next(r for r in rows if r['image_id'] == sample['image_id'])
            hashes = sample['source_hashes']
            require(row['source_hashes'] == {'rgb': hashes['image_sha256'],
                    'annotation': hashes['annotation_sha256']}, 'Receipt source pins mismatch')
            if kind == 'rights':
                require(row['oam_id'] == sample['oam_id'] and row['source_id_match'] is True
                        and row['metadata_object_identity_match'] is True
                        and row['stac']['sha256'] == sample['stac_sha256'],
                        'Rights identity mismatch')
                require(all(row[k] == 'CC-BY-4.0' for k in
                        ('image_license', 'annotation_license', 'original_source_license')),
                        'Rights declarations incomplete')
            if kind == 'independence':
                require(row['split'] == sample['split'] and row['fold'] == sample['validation_fold']
                        and row['original_source_id'] == sample['original_source_id']
                        and row['oam_id'] == sample['oam_id'], 'Independence identity mismatch')
                require(row['guard']['proximity_hold_m'] == 1000
                        and row['guard']['nearest_opposite_footprint_lower_bound_m'] >= 1000,
                        'Proximity guard unresolved')
            if kind == 'coverage':
                require(row['formal_validity_assertion'] == 'NOT_ISSUED'
                        and row['new_major_suspicion'] is False
                        and row['all_valid_certified'] is False,
                        'Coverage contradiction requires review')
    for sample in package['samples']:
        for field, hash_key in [('rgb', 'image_sha256'), ('annotation', 'annotation_sha256')]:
            name = sample['source_files'][field].removeprefix('extracted/')
            member = next((m for m in members if m['member'] == name), None)
            require(member is not None and member['equal'] is True and member['member_bytes'] > 0
                    and member['member_sha256'] == member['local_sha256']
                    == sample['source_hashes'][hash_key], 'Archive member mismatch')
    return {key: reference['sha256'] for key, reference in package['evidence'].items()}


def check_readiness(package_path, package_sha256, catalog_path, evidence_root=None):
    package = checked_snapshot(package_path, package_sha256)
    require(package['schema'] == 'ouem-r1d-four-readiness-v1'
            and package['mode'] == 'EVIDENCE_ONLY', 'Evidence-only schema required')
    require(package['source_generation'] == GENERATION, 'Source generation mismatch')
    require(package['execution_enabled'] is False and package['formal_r1'] == {'HOLD': 150},
            'Execution or acceptance upgrade forbidden')
    require(package['human_authorization'] == {'status': 'PENDING', 'reference': None, 'date': None},
            'Human authorization must remain pending')
    require(package['decisions']['G3']['status'] == package['decisions']['G4']['status'] == 'PROPOSED'
            and package['decisions']['G3']['scope'] == 'official_holdout_and_unlisted_neighbor_sources'
            and package['decisions']['G4']['operational_assertion'] == 'REVIEW_PASS'
            and package['decisions']['G4']['ALL_PIXELS_PROVEN_VALID'] == 'NOT_VERIFIED'
            and package['decisions']['G4']['adapter_token'] == 'ALL_VALID_REQUIRES_EXPLICIT_HUMAN_DECISION',
            'Decision upgrade forbidden')
    require(package['resources'] == dict(LIMITS, network_requests=0, network_bytes=0),
            'Resource contract mismatch')
    require(package['offline_boundary']['status'] == 'NOT_VERIFIED', 'Isolation is unproved')
    require(package['semantic_label']['categories'] == [1, 2]
            and package['semantic_label']['policy'] == 'upstream_nonzero_binary_with_review'
            and package['semantic_label']['rule'] == 'np.any(panoptic_rgb !=0,axis=2).astype(uint8)'
            and package['semantic_label']['per_segment_category_status'] == 'NOT_VERIFIED'
            and package['semantic_label']['uniform_masks'] == 'PRESERVE', 'Label policy mismatch')
    samples = package['samples']
    require([s['image_id'] for s in samples] == list(TARGETS), 'Fixed four order required')
    for sample, oam, size, fold in zip(samples, OAMS, SHAPES, FOLDS):
        require(sample['oam_id'] == oam and sample['split'] == TARGETS[sample['image_id']]
                and sample['validation_fold'] == fold and sample['expected_shape'] == [size, size]
                and sample['source_generation'] == GENERATION and sample['formal_r1'] == 'HOLD'
                and sample['technical'] == 'PASS',
                'Sample identity/grid mismatch')
        for value in sample['source_hashes'].values():
            require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid source SHA')
        require(set(sample['source_hashes']) == {'image_sha256', 'annotation_sha256'},
                'Missing source pins')
        require(set(sample['observed_categories']) <= {1, 2}, 'Unknown category')
        for field, folder, suffix in [('rgb', 'images', 'tif'), ('annotation', 'masks', 'png')]:
            require(sample['source_files'][field] ==
                    f'extracted/dataset/holdout/{folder}/{oam}_{sample["image_id"]}.{suffix}',
                    'Source filename mismatch')
    require(set(package['evidence']) == {'review', 'archive', 'rights', 'independence', 'coverage'},
            'Missing receipt reference')
    for reference in package['evidence'].values():
        require(relative_path(reference['path']).endswith('.json')
                and re.fullmatch('[0-9a-f]{64}', reference['sha256']), 'Invalid receipt reference')
    catalog = checked_snapshot(catalog_path, CATALOG_SHA)
    canonical = json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode('utf-8')
    require(digest(canonical) == CATALOG_SHA and package['catalog']['sha256'] == CATALOG_SHA
            and package['catalog']['canonical_bytes'] == len(canonical)
            and package['catalog']['materialized'] is True, 'Catalog canonical bytes mismatch')
    verified = check_receipts(package, catalog, evidence_root) if evidence_root is not None else {}
    return dict(status='EVIDENCE_PACKAGE_CHECKED_EXECUTION_BLOCKED',
                package_sha256=package_sha256, catalog_sha256=CATALOG_SHA,
                receipts_checked=verified, source_bytes_checked=False,
                g3_g4_approval='PENDING', isolation='NOT_VERIFIED',
                execution_enabled=False, formal_r1={'HOLD': 150})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--package-sha256', required=True)
    parser.add_argument('--catalog-index', required=True, type=Path)
    parser.add_argument('--evidence-root', type=Path)
    args = parser.parse_args()
    try:
        result = check_readiness(args.package, args.package_sha256, args.catalog_index, args.evidence_root)
    except (Stop, ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
        print(json.dumps(dict(status='STOP', reason=str(exc), execution_enabled=False)))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
