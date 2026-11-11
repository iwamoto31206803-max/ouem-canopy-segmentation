"""Evidence-only checks use JSON fixtures, never image inputs or network."""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import r1d_real_readiness as readiness


DOCS = Path(__file__).parents[1] / 'docs' / 'implementation'


@pytest.fixture
def package():
    return json.loads((DOCS / 'R1D1B6_DECISION_PACKAGE.json').read_text(encoding='utf-8'))


def seal(tmp_path, package):
    path = tmp_path / 'package.json'
    data = json.dumps(package).encode()
    path.write_bytes(data)
    return path, readiness.digest(data)


def check(tmp_path, package, **kwargs):
    path, sha = seal(tmp_path, package)
    return readiness.check_readiness(path, sha, DOCS / 'R1D1B6_CATALOG_INDEX.json', **kwargs)


def receipts(package):
    members = []
    rights = []
    independent = []
    coverage = []
    for sample in package['samples']:
        hashes = sample['source_hashes']
        pins = dict(rgb=hashes['image_sha256'], annotation=hashes['annotation_sha256'])
        for field, hash_key in [('rgb', 'image_sha256'), ('annotation', 'annotation_sha256')]:
            members.append(dict(member=sample['source_files'][field].removeprefix('extracted/'),
                                member_bytes=1, member_sha256=hashes[hash_key],
                                local_sha256=hashes[hash_key], equal=True))
        rights.append(dict(image_id=sample['image_id'], source_hashes=pins,
                           oam_id=sample['oam_id'], source_id_match=True,
                           metadata_object_identity_match=True,
                           stac=dict(sha256=sample['stac_sha256']), image_license='CC-BY-4.0',
                           annotation_license='CC-BY-4.0', original_source_license='CC-BY-4.0'))
        independent.append(dict(image_id=sample['image_id'], source_hashes=pins,
                                oam_id=sample['oam_id'], split=sample['split'],
                                fold=sample['validation_fold'], original_source_id=sample['original_source_id'],
                                guard=dict(proximity_hold_m=1000,
                                           nearest_opposite_footprint_lower_bound_m=1001)))
        coverage.append(dict(image_id=sample['image_id'], source_hashes=pins,
                             formal_validity_assertion='NOT_ISSUED', new_major_suspicion=False,
                             all_valid_certified=False))
    catalog = json.loads((DOCS / 'R1D1B6_CATALOG_INDEX.json').read_text(encoding='utf-8'))
    return dict(review={'synthetic_fixture': True},
                archive=dict(complete=True, source_generation=readiness.GENERATION,
                             archives=[dict(checksum_match=True, gzip_stream_integrity=True,
                                            normalized_paths_unique=True, target_members=members)]),
                rights=dict(images=rights), independence=dict(hard_stop=False, results=independent,
                    catalog_sha256={item['name']: item['sha256'] for item in catalog['metadata']}),
                coverage=dict(images=coverage))


def write_receipts(tmp_path, package, values):
    root = tmp_path / 'receipts'
    root.mkdir()
    for key, value in values.items():
        data = json.dumps(value).encode()
        (root / f'{key}.json').write_bytes(data)
        package['evidence'][key] = dict(path=f'{key}.json', sha256=readiness.digest(data))
    return root


def test_package_only_is_always_blocked(tmp_path, package, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Image or converter accessed')
    import rasterio
    from PIL import Image
    import m1_5d_zenodo_prepare
    monkeypatch.setattr(Image, 'open', forbidden)
    monkeypatch.setattr(rasterio, 'open', forbidden)
    monkeypatch.setattr(m1_5d_zenodo_prepare, 'convert', forbidden)
    before = set(tmp_path.iterdir())
    result = check(tmp_path, package)
    assert result['execution_enabled'] is False
    assert result['source_bytes_checked'] is False
    assert result['receipts_checked'] == {}
    assert result['status'] == 'PACKAGE_SCHEMA_CHECKED_EXECUTION_BLOCKED'
    assert result['formal_r1'] == {'HOLD': 150}
    assert set(tmp_path.iterdir()) == before | {tmp_path / 'package.json'}


def test_synthetic_receipts_check_without_approval(tmp_path, package):
    root = write_receipts(tmp_path, package, receipts(package))
    result = check(tmp_path, package, evidence_root=root)
    assert len(result['receipts_checked']) == 5
    assert result['status'] == 'EVIDENCE_PACKAGE_CHECKED_EXECUTION_BLOCKED'
    assert result['g3_g4_approval'] == 'PENDING'
    assert not result['execution_enabled']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(source_generation='hf'),
    lambda p: p.update(execution_enabled=True),
    lambda p: p['formal_r1'].update(HOLD=149),
    lambda p: p['human_authorization'].update(status='APPROVED'),
    lambda p: p['decisions']['G3'].update(status='APPROVED'),
    lambda p: p['decisions']['G3'].update(scope='within_manifest_only'),
    lambda p: p['decisions']['G4'].update(operational_assertion='ALL_VALID'),
    lambda p: p['decisions']['G4'].update(ALL_PIXELS_PROVEN_VALID='PROVEN'),
    lambda p: p['decisions']['G4'].update(adapter_token='ALL_VALID'),
    lambda p: p['resources'].update(network_bytes=1),
    lambda p: p['resources'].update(memory_bytes=3*1024**3),
    lambda p: p['offline_boundary'].update(status='VERIFIED'),
    lambda p: p['samples'][0].update(split='val'),
    lambda p: p['samples'][0].update(validation_fold=0),
    lambda p: p['samples'][0].update(expected_shape=[391, 391]),
    lambda p: p['samples'][0].update(oam_id='other'),
    lambda p: p['samples'].reverse(),
    lambda p: p['samples'][0]['source_files'].update(rgb='../source.tif'),
    lambda p: p['samples'][0]['source_hashes'].update(image_sha256='bad'),
    lambda p: p['samples'][0].update(observed_categories=[3]),
    lambda p: p['semantic_label'].update(per_segment_category_status='VERIFIED'),
    lambda p: p['semantic_label'].update(rule='replace all background'),
    lambda p: p['evidence'].pop('coverage'),
    lambda p: p['evidence']['review'].update(path='../review.json'),
    lambda p: p['evidence']['review'].update(path='source.tif'),
    lambda p: p['catalog'].update(materialized=False),
])
def test_invalid_decision_or_input_binding(tmp_path, package, mutation):
    mutation(package)
    with pytest.raises(readiness.Stop):
        check(tmp_path, package)


def test_package_and_catalog_hash_rejection(tmp_path, package):
    path, sha = seal(tmp_path, package)
    with pytest.raises(readiness.Stop, match='SHA'):
        readiness.check_readiness(path, '0'*64, DOCS / 'R1D1B6_CATALOG_INDEX.json')
    catalog = tmp_path / 'catalog.json'
    catalog.write_bytes((DOCS / 'R1D1B6_CATALOG_INDEX.json').read_bytes() + b'\n')
    with pytest.raises(readiness.Stop, match='SHA'):
        readiness.check_readiness(path, sha, catalog)


@pytest.mark.parametrize('kind', ['rights', 'independence', 'coverage', 'archive'])
def test_receipt_pin_or_identity_mismatch(tmp_path, package, kind):
    values = receipts(package)
    if kind == 'archive':
        values[kind]['archives'][0]['target_members'][0]['local_sha256'] = '0'*64
    else:
        key = 'results' if kind == 'independence' else 'images'
        values[kind][key][0]['source_hashes']['rgb'] = '0'*64
    root = write_receipts(tmp_path, package, values)
    with pytest.raises(readiness.Stop, match='mismatch'):
        check(tmp_path, package, evidence_root=root)


def test_receipt_byte_mutation_and_hard_stop(tmp_path, package):
    values = receipts(package)
    values['independence']['hard_stop'] = True
    root = write_receipts(tmp_path, package, values)
    with pytest.raises(readiness.Stop, match='HARD STOP'):
        check(tmp_path, package, evidence_root=root)
    (root / 'review.json').write_text('{}', encoding='utf-8')
    with pytest.raises(readiness.Stop, match='SHA'):
        check(tmp_path, package, evidence_root=root)


def test_cli_cannot_execute_and_returns_stop(tmp_path, package):
    path, sha = seal(tmp_path, package)
    command = [sys.executable, '-B', str(Path(readiness.__file__)), '--package', str(path),
               '--package-sha256', sha, '--catalog-index', str(DOCS / 'R1D1B6_CATALOG_INDEX.json')]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['execution_enabled'] is False
    assert json.loads(result.stdout)['status'] == 'PACKAGE_SCHEMA_CHECKED_EXECUTION_BLOCKED'
    assert subprocess.run(command + ['--execute'], capture_output=True).returncode == 2
    command[command.index(sha)] = '0'*64
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['status'] == 'STOP'


def test_receipt_reparse_and_size_guard(tmp_path, package, monkeypatch):
    root = write_receipts(tmp_path, package, receipts(package))
    def reparse(path):
        raise readiness.Stop('Symlink/reparse point forbidden')
    monkeypatch.setattr(readiness, 'ordinary', reparse)
    with pytest.raises(readiness.Stop, match='reparse'):
        check(tmp_path, package, evidence_root=root)
    monkeypatch.undo()
    (root / 'review.json').write_bytes(b' ' * (4 * 1024**2 + 1))
    with pytest.raises(readiness.Stop, match='ceiling'):
        check(tmp_path, package, evidence_root=root)


@pytest.mark.parametrize('data', [b'{"key":0,"key":1}', b'{"key":NaN}'])
def test_snapshot_rejects_ambiguous_json(tmp_path, data):
    path = tmp_path / 'bad.json'
    path.write_bytes(data)
    with pytest.raises(readiness.Stop):
        readiness.checked_snapshot(path, readiness.digest(data))


@pytest.mark.parametrize('kind', ['catalog', 'proximity', 'coverage', 'rights'])
def test_finite_evidence_contradictions_stop(tmp_path, package, kind):
    values = receipts(package)
    if kind == 'catalog':
        values['independence']['catalog_sha256']['train_meta.json'] = '0'*64
    elif kind == 'proximity':
        values['independence']['results'][0]['guard']['nearest_opposite_footprint_lower_bound_m'] = 999
    elif kind == 'coverage':
        values['coverage']['images'][0]['new_major_suspicion'] = True
    else:
        values['rights']['images'][0]['original_source_license'] = 'unknown'
    root = write_receipts(tmp_path, package, values)
    with pytest.raises(readiness.Stop):
        check(tmp_path, package, evidence_root=root)


@pytest.mark.parametrize('sample_index', range(4))
@pytest.mark.parametrize('field', ['image_id', 'validation_fold', 'height', 'width'])
def test_float_equal_to_expected_integer_rejected(tmp_path, package, sample_index, field):
    sample = package['samples'][sample_index]
    if field in ('height', 'width'):
        coordinate = 0 if field == 'height' else 1
        sample['expected_shape'][coordinate] = float(sample['expected_shape'][coordinate])
    else:
        sample[field] = float(sample[field])
    # seal() supplies a matching new file hash, so type validation must reject it.
    with pytest.raises(readiness.Stop, match='integer|Integer'):
        check(tmp_path, package)


@pytest.mark.parametrize('value', [False, True, '0', None])
def test_noninteger_fold_rejected(tmp_path, package, value):
    package['samples'][2]['validation_fold'] = value
    with pytest.raises(readiness.Stop, match='Integer'):
        check(tmp_path, package)


@pytest.mark.parametrize('field', ['image_id', 'height', 'width'])
@pytest.mark.parametrize('value', [True, '717', None])
def test_noninteger_identity_or_shape_rejected(tmp_path, package, field, value):
    if field == 'image_id':
        package['samples'][0][field] = value
    else:
        package['samples'][0]['expected_shape'][0 if field == 'height' else 1] = value
    with pytest.raises(readiness.Stop, match='integer|Integer'):
        check(tmp_path, package)


def test_float_catalog_byte_count_rejected(tmp_path, package):
    package['catalog']['canonical_bytes'] = float(package['catalog']['canonical_bytes'])
    with pytest.raises(readiness.Stop, match='Integer catalog'):
        check(tmp_path, package)


@pytest.mark.parametrize('field', ['oam_groups', 'rows', 'reviewed_subject_ids'])
@pytest.mark.parametrize('value_type', [float, bool])
def test_catalog_numerical_types_rejected(tmp_path, package, monkeypatch, field, value_type):
    catalog = json.loads((DOCS / 'R1D1B6_CATALOG_INDEX.json').read_text(encoding='utf-8'))
    if field == 'rows':
        catalog['metadata'][0]['rows'] = value_type(catalog['metadata'][0]['rows'])
    elif field == 'reviewed_subject_ids':
        catalog[field][0] = value_type(catalog[field][0])
    else:
        catalog[field] = value_type(catalog[field])
    data = json.dumps(catalog, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    catalog_path = tmp_path / 'synthetic-catalog.json'
    catalog_path.write_bytes(data)
    sha = readiness.digest(data)
    # A synthetic catalog pin exercises type checks beyond the production hash gate.
    monkeypatch.setattr(readiness, 'CATALOG_SHA', sha)
    package['catalog'].update(sha256=sha, canonical_bytes=len(data))
    path, package_sha = seal(tmp_path, package)
    with pytest.raises(readiness.Stop, match='Integer catalog'):
        readiness.check_readiness(path, package_sha, catalog_path)


@pytest.mark.parametrize('kind', ['rights', 'independence', 'coverage', 'fold'])
def test_receipt_integer_identity_required(tmp_path, package, kind):
    values = receipts(package)
    if kind == 'fold':
        values['independence']['results'][2]['fold'] = 0.0
    else:
        key = 'results' if kind == 'independence' else 'images'
        values[kind][key][0]['image_id'] = 717.0
    root = write_receipts(tmp_path, package, values)
    with pytest.raises(readiness.Stop):
        check(tmp_path, package, evidence_root=root)


def test_cli_full_receipt_status(tmp_path, package):
    root = write_receipts(tmp_path, package, receipts(package))
    path, sha = seal(tmp_path, package)
    result = subprocess.run([sys.executable, '-B', str(Path(readiness.__file__)),
        '--package', str(path), '--package-sha256', sha,
        '--catalog-index', str(DOCS / 'R1D1B6_CATALOG_INDEX.json'),
        '--evidence-root', str(root)], capture_output=True, text=True)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report['status'] == 'EVIDENCE_PACKAGE_CHECKED_EXECUTION_BLOCKED'
    assert len(report['receipts_checked']) == 5
    assert report['execution_enabled'] is False


def test_documented_expected_package_hash_is_independently_pinned():
    document = (DOCS / 'R1D1B6_REAL_READINESS.md').read_text(encoding='utf-8')
    expected = re.search(r"\$ExpectedPackageSHA = '([0-9a-f]{64})'", document).group(1)
    assert expected == readiness.digest((DOCS / 'R1D1B6_DECISION_PACKAGE.json').read_bytes())
    assert '--package-sha256 $ExpectedPackageSHA' in document
    assert 'if ($MeasuredPackageSHA -ne $ExpectedPackageSHA)' in document
