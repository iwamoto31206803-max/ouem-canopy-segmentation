"""Offline M1-2D source geometry gate for the two approved cached pairs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import rasterio
import rasterio.warp
from pyproj import Geod, Transformer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(crs, x, y, dx, dy):
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = inverse.transform(x, y)
    geod = Geod(ellps='WGS84')
    return dict(latitude=lat, longitude=lon,
                east_m=geod.inv(lon, lat, *inverse.transform(x + dx, y))[2],
                north_m=geod.inv(lon, lat, *inverse.transform(x, y + dy))[2])


def require_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch for {path.name}')
    return actual


def audit(cache, manifest):
    if {s['image_id'] for s in manifest['samples']} != {661, 1972}:
        raise ValueError('Only the two approved samples are supported')
    groups = {}
    results = []
    for sample in manifest['samples']:
        group, split = sample['oam_id'], sample['split']
        expected_split = 'train' if sample['image_id'] == 661 else 'val'
        if split != expected_split or (group in groups and groups[group] != split):
            raise ValueError('Split/source-group inconsistency')
        groups[group] = split
        folder = cache / str(sample['image_id'])
        hashes = {name: require_hash(folder / name, expected) for name, expected in sample['hashes'].items()}
        stac_path = cache / (group + '-stac.json')
        stac_hash = require_hash(stac_path, sample['source_stac_sha256'])
        stac = json.loads(stac_path.read_text(encoding='utf-8-sig'))
        if stac['id'] != group or stac['properties']['license'].lower() != 'cc-by-4.0':
            raise ValueError('STAC identity/license mismatch')
        with rasterio.open(folder / 'rgb.bin') as src:
            if src.crs is None or src.width != 2048 or src.height != 2048:
                raise ValueError('Unexpected source geometry')
            if src.transform.b or src.transform.d:
                raise ValueError('Rotated source grid requires review')
            x, y = src.xy(src.height // 2, src.width // 2)
            source = intervals(src.crs, x, y, *src.res)
            bounds_wgs84 = rasterio.warp.transform_bounds(src.crs, 4326, *src.bounds)
            bbox = stac['bbox']
            contained = (bbox[0] <= bounds_wgs84[0] <= bounds_wgs84[2] <= bbox[2]
                         and bbox[1] <= bounds_wgs84[1] <= bounds_wgs84[3] <= bbox[3])
            valid = bool((src.dataset_mask() == 255).all())
            results.append(dict(image_id=sample['image_id'], oam_id=group, split=split,
                source_hashes=hashes, stac_sha256=stac_hash, source_crs=str(src.crs),
                source_bounds=list(src.bounds), source_transform=list(src.transform)[:6],
                input_shape=[src.height, src.width], projected_pixel_m=list(src.res),
                local_ground_pixel=source, source_wgs84_bounds=list(bounds_wgs84),
                contained_in_stac_bbox=contained, nodata=src.nodata, gdal_all_valid=valid,
                proposed_crop=[0, 0, 2040, 2040], proposed_output_shape=[408,408],
                nominal_gsd_m=0.5, fivefold_ground_pixel_m=[source['east_m']*5,source['north_m']*5],
                rights_evidence='M1_2B verified; user conditionally accepted assigned split',
                attribution_reference=sample['source_stac_url'],
                status='HOLD_GSD_MISMATCH', training_pair_generated=False))
    return dict(utc=datetime.now(timezone.utc).isoformat(), dataset=manifest['dataset'],
                revision=manifest['revision'], status='HOLD_GSD_MISMATCH', samples=results,
                target_reference_m=[0.484,0.486], output_hashes={},
                reason='Fivefold projected-grid reduction differs materially from target ground spacing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path('docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())
    print(json.dumps(audit(args.cache, manifest), indent=2))
