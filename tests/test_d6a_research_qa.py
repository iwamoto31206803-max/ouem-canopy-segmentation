import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import d6a_research_qa as qa


@pytest.fixture
def fixture(tmp_path):
    root, evidence = tmp_path / 'cache', tmp_path / 'stac'
    evidence.mkdir()
    meta = {key: root / value for key, value in qa.FILES.items()}
    meta['coco'].parent.mkdir(parents=True)
    images = root / 'extracted/dataset/holdout/images'
    masks = images.with_name('masks')
    images.mkdir(parents=True)
    masks.mkdir()
    rows, candidates, annotations = [], [], []
    def add(i, *, fold=1, group=None, x=None, rgb='normal', label=1, nodata=None, internal_mask=False):
        group = group or f'source-{i}'
        x = i * 15000 if x is None else x
        row = dict(image_id=i, oam_id=group, validation_fold=fold, license='CC-BY 4.0',
                   width=256, height=256, crs='EPSG:3395', bounds=[x, 0, x+128, 128],
                   meta=f'https://example.org/stac/{group}.json')
        rows.append(row)
        if fold < 0:
            return row
        candidate = {k: row[k] for k in ('image_id', 'oam_id', 'validation_fold')}
        candidates.append(candidate)
        if label:
            annotations.append(dict(id=i, image_id=i, category_id=1, iscrowd=0))
        ring = [[-1, -1], [10, -1], [10, 1], [-1, 1], [-1, -1]]
        item = dict(type='Feature', stac_version='1.0.0', id=group, bbox=[-1, -1, 10, 1],
                    geometry=dict(type='Polygon', coordinates=[ring]),
                    properties={'license': 'cc-by-4.0', 'oam:producer_name': 'Test',
                                'providers': [{'name': 'Test', 'roles': ['licensor']}]},
                    assets={'metadata': {'href': row['meta']}, 'visual': {'href': 'https://example.org/image'}})
        stac = evidence / f'{group}-stac.json'
        stac.write_text(json.dumps(item))
        candidate['stac_sha256'] = qa.sha(stac)
        count = {'gray': 1, 'alpha': 4}.get(rgb, 3)
        pixels = np.full((count, 256, 256), 100, dtype=np.uint8)
        if rgb == 'inside':
            pixels[:, 80:110, 80:110] = 0
            pixels[:, 120:175, 120:175] = 255
        if rgb in ('edge', 'blocked'):
            pixels[:, :, :20 if rgb == 'edge' else 45] = 0
        image, mask = images / f'{group}_{i}.tif', masks / f'{group}_{i}.png'
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(image, 'w', driver='GTiff', width=256, height=256, count=count,
                               dtype='uint8', crs=row['crs'], transform=from_origin(x, 128, .5, .5),
                               nodata=nodata) as dst:
                dst.write(pixels)
                if internal_mask:
                    valid = np.full((256, 256), 255, np.uint8)
                    valid[0, 0] = 0
                    dst.write_mask(valid)
        Image.fromarray(np.full((256, 256, 3), label, dtype=np.uint8)).save(mask)
        candidate['source_hashes'] = dict(rgb=qa.sha(image), annotation=qa.sha(mask))
        return row
    add(50, fold=-1)
    path = tmp_path / 'policy.json'
    def run(reverse=False):
        for name, role in [('train', lambda r: r['validation_fold'] >= 0), ('test', lambda r: r['validation_fold'] < 0)]:
            meta[name].write_text(''.join(json.dumps(r)+'\n' for r in rows if role(r)))
        meta['coco'].write_text(json.dumps(dict(annotations=annotations)))
        path.write_text(json.dumps(dict(schema=qa.POLICY, source_generation=qa.GENERATION, seed=20,
                                        catalog_sha256={k: qa.sha(v) for k, v in meta.items()},
                                        candidates=candidates[::-1] if reverse else candidates)))
        return qa.assess(root, evidence, path, qa.sha(path))
    return add, candidates, annotations, run, root, evidence, path


def decision(item):
    result = item['final_decision']
    return result['status'], result['reason']


def test_upstream_unmodified_labels_and_black_geometry(fixture):
    add, _, _, run, *_ = fixture
    add(1, rgb='inside', label=0)
    add(2, rgb='edge', label=1)
    add(3, rgb='blocked', label=1)
    report = run()
    bg, edge, blocked = report['results']
    assert decision(bg) == decision(edge) == (qa.ELIGIBLE, 'CLEAN_224_WINDOW')
    assert bg['crop_binary_fraction'] == 0
    assert edge['crop_binary_fraction'] == 1
    assert bg['border_black_pixels'] == 0
    assert edge['border_black_pixels'] == 20 * 256
    assert edge['crop_xywh'][0] >= 20 and edge['selected_black_support_cells'] == 0
    assert decision(blocked) == (qa.HOLD, 'NO_CLEAN_224_WINDOW')
    assert blocked['source_decision']['status'] == qa.ELIGIBLE
    assert report['eligible_ids'] == {'train': [1, 2], 'val': []}
    assert report['counts'] == {qa.ELIGIBLE: 2, qa.HOLD: 1}
    assert report['labels'] == qa.LABEL_POLICY and report['formal_r1'] == 'UNCHANGED'
    assert run(reverse=True)['results'] == run()['results'] == report['results']


@pytest.mark.parametrize('case,expected', [
    ('missing_stac', (qa.HOLD, 'STAC_EVIDENCE_MISSING')),
    ('missing_stac_pin', (qa.HOLD, 'STAC_PIN_MISSING')),
    ('conflicting_stac', (qa.HOLD, 'STAC_DECLARATION_CONFLICT')),
    ('stac_hash', (qa.FAIL, 'STAC_HASH_MISMATCH')),
    ('unknown_license', (qa.HOLD, 'CC_BY_4_0_DECLARATION_REQUIRED')),
    ('fold', (qa.HOLD, 'CANDIDATE_IDENTITY_CONFLICT')),
    ('nodata', (qa.HOLD, 'DECLARED_NODATA_ALPHA_OR_INVALID_MASK')),
    ('mask', (qa.HOLD, 'DECLARED_NODATA_ALPHA_OR_INVALID_MASK')),
    ('alpha', (qa.HOLD, 'DECLARED_ALPHA_OR_TRANSPARENCY')),
    ('gray', (qa.FAIL, 'RGB_UINT8_ENCODING')),
    ('dimensions', (qa.FAIL, 'IMAGE_MASK_METADATA_DIMENSIONS')),
    ('bounds', (qa.FAIL, 'GRID_CRS_OR_METADATA_BOUNDS')),
    ('hash', (qa.FAIL, 'SOURCE_HASH_MISMATCH')),
    ('category', (qa.FAIL, 'LABEL_ENCODING_OR_CATEGORY_CONFLICT')),
])
def test_machine_reasons(fixture, case, expected):
    add, candidates, annotations, run, _, evidence, _ = fixture
    row = add(1, rgb=case if case in ('alpha', 'gray') else 'normal',
              nodata=0 if case == 'nodata' else None, internal_mask=case == 'mask')
    c, stac = candidates[0], evidence / 'source-1-stac.json'
    if case == 'missing_stac':
        stac.unlink()
    elif case == 'missing_stac_pin':
        del c['stac_sha256']
    elif case in ('conflicting_stac', 'stac_hash'):
        item = json.loads(stac.read_bytes())
        item['properties']['license'] = 'cc-by-nc-4.0'
        stac.write_text(json.dumps(item))
        if case == 'conflicting_stac':
            c['stac_sha256'] = qa.sha(stac)
    elif case == 'unknown_license':
        row['license'] = None
    elif case == 'fold':
        c['validation_fold'] = 0
    elif case == 'dimensions':
        row['height'] = 255
    elif case == 'bounds':
        row['bounds'][2] += 1
    elif case == 'hash':
        c['source_hashes']['rgb'] = '0'*64
    elif case == 'category':
        annotations[0]['category_id'] = 3
    assert decision(run()['results'][0]) == expected


def test_catalog_split_groups_and_holdout_excluded_before_source_io(fixture):
    add, candidates, _, run, *_ = fixture
    add(1, group='shared')
    add(2, group='shared')
    add(3, x=750000)  # Overlaps official Test source 50.
    add(4, fold=0)
    add(5, fold=1, x=60000)  # Overlaps Val source 4.
    candidates.append(dict(image_id=50, oam_id='source-50', validation_fold=-1))
    results = {r['image_id']: decision(r) for r in run()['results']}
    assert results[1] == results[2] == (qa.EXCLUDE, 'DUPLICATE_POOL_SOURCE_GROUP')
    assert results[3] == results[4] == results[5] == (qa.HOLD, 'HOLD_CROSS_SPLIT_FOOTPRINT')
    assert results[50] == (qa.EXCLUDE, 'OFFICIAL_TEST_OR_HOLDOUT')


def test_cross_split_original_metadata_alias(fixture):
    add, _, _, run, *_ = fixture
    train = add(1)
    val = add(2, fold=0)
    val['meta'] = train['meta'].replace('example.org', 'other.example')
    assert {decision(entry) for entry in run()['results']} == {(qa.HOLD, 'CROSS_SPLIT_SOURCE_IDENTITY')}


def test_observed_and_expected_source_hashes_have_distinct_assurance(fixture):
    add, candidates, _, run, *_ = fixture
    add(1)
    pinned = run()['results'][0]
    assert pinned['source_hash_verification'] == 'MATCHED_EXTERNAL_EXPECTED_SHA256'
    assert pinned['expected_source_hashes'] == pinned['source_hashes']
    expected = candidates[0].pop('source_hashes')
    observed = run()['results'][0]
    assert observed['source_hashes'] == expected
    assert observed['source_hash_verification'] == 'SOURCE_AUTHENTICITY_NOT_EXTERNALLY_VERIFIED'
    assert 'expected_source_hashes' not in observed
    assert observed['final_decision']['status'] == qa.ELIGIBLE
    candidates[0]['source_hashes'] = dict(expected, rgb='0'*64)
    mismatch = run()['results'][0]
    assert mismatch['source_hash_verification'] == 'EXTERNAL_EXPECTED_SHA256_MISMATCH'
    assert decision(mismatch) == (qa.FAIL, 'SOURCE_HASH_MISMATCH')


def test_missing_window_fails_closed():
    assert qa.final_decision({'status': qa.ELIGIBLE}, None)['status'] == qa.FAIL


def test_report_environment(fixture):
    add, _, _, run, *_ = fixture
    add(1)
    env = run()['environment']
    assert qa.digest(env['script_sha256_raw'])
    assert len(env['git_commit']) == 40 and isinstance(env['git_dirty'], bool)
    assert set(env['dependencies']) == {'numpy', 'Pillow', 'rasterio', 'pyproj'}
    assert all(env[k] for k in ('python', 'os', 'gdal'))


def test_manifest_pin_and_exclusive_cli_output(fixture, tmp_path):
    add, _, _, run, root, evidence, contract = fixture
    add(1)
    run()
    out = tmp_path / 'report.json'
    args = ['--root', str(root), '--evidence', str(evidence), '--contract', str(contract), '--output', str(out)]
    with pytest.raises(SystemExit):
        qa.main(args + ['--contract-sha256', '0'*64])
    assert not out.exists()
    args += ['--contract-sha256', qa.sha(contract)]
    assert qa.main(args) == 0
    original = out.read_bytes()
    with pytest.raises(SystemExit):
        qa.main(args)
    assert out.read_bytes() == original
