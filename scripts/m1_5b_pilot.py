"""Bounded OAM-TCD pilot: plan/acquire online; prepare/verify strictly offline."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import struct
import time
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from pyproj import Geod
import rasterio
import rasterio.warp
import requests

from m1_2a_data_qa import REVISION, CODE_REVISION, ROOT, labels, save_pair
from m1_2b_overlay_qa import overlay, decode_rle
from m1_2d_geometry_audit import sha, require_hash, intervals
from m1_2d_prepare_pairs import output_grid, reduce_area, binary
from m1_5a_inventory import role, overlaps, split_audit, verify_existing

REPO = Path(__file__).resolve().parents[1]
ALLOWED = {2, 3, 4, 5, 6, 11, 17, 9, 576, 1401, 4235}
MIB = 1024**2
COLUMNS = ['image_id', 'oam_id', 'license', 'validation_fold', 'height', 'width',
           'crs', 'bounds', 'meta', 'image', 'annotation', 'segments', 'coco_annotations']
PRODUCTS = {'rgb.tif', 'canopy_fraction.tif', 'canopy_mask.tif', 'valid_fraction.tif'}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fresh(path, protected):
    path = path.resolve()
    if path == REPO or REPO in path.parents or path.exists():
        raise ValueError('Use a fresh external destination')
    if any(path == p.resolve() or p.resolve() in path.parents or path in p.resolve().parents for p in protected):
        raise ValueError('Destination overlaps protected inputs')
    path.mkdir(parents=True, exist_ok=False)


def stored(*roots):
    return sum(p.stat().st_size for root in roots if root and root.exists() for p in root.rglob('*') if p.is_file())


def storage_gate(cache, output, extra, limit):
    # Reserve 256 KiB for small reports even if a data write is rejected.
    if stored(cache, output) + extra > limit - 256*1024:
        raise ValueError('Storage budget exceeded (including new range cache and products)')


def load_inventory(cache, ids):
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= ALLOWED:
        raise ValueError('Only explicit M1-5A candidates allowed; image 1 and existing/holdout IDs excluded')
    reference = json.loads((REPO/'docs/implementation/M1_5A_INVENTORY_RESULT.json').read_text())
    require_hash(cache/'rows.json', reference['metadata_cache_sha256'])
    acquisition = json.loads((cache/'acquisition.json').read_text())
    if acquisition['revision'] != REVISION or acquisition['shards'] != reference['acquisition']['shards']:
        raise ValueError('Inventory revision/shard provenance mismatch')
    rows = json.loads((cache/'rows.json').read_text())
    split_audit(rows)
    verify_existing(rows, json.loads((REPO/'docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())['samples'])
    indexed = {r['image_id']: r for r in rows}
    candidates = {r['image_id']: r for r in reference['candidates']}
    selected = []
    for rid in ids:
        row = indexed[rid]
        if any(row[k] != candidates[rid][k] for k in row):
            raise ValueError('Candidate differs from pinned inventory')
        selected.append(dict(candidates[rid]))
    return selected, rows, reference


def separation(row, rows):
    if row['license'] != 'CC-BY 4.0' or role(row) not in ('train', 'val') or row['official_split'] != 'train':
        raise ValueError('HOLD_LICENSE_OR_SPLIT')
    geod = Geod(ellps='WGS84')
    def radius(r):
        b = r['wgs84_bounds']; c = r['center']
        return max(geod.inv(*c, x, y)[2] for x in (b[0], b[2]) for y in (b[1], b[3]))
    distances = []
    for other in rows:
        if other['image_id'] == row['image_id'] or role(other) == role(row): continue
        if other['oam_id'] == row['oam_id']: raise ValueError('HOLD_CROSS_SPLIT_OAM')
        if row.get('meta') and other.get('meta') and urlparse(row['meta']).path == urlparse(other['meta']).path:
            raise ValueError('HOLD_CROSS_SPLIT_ORIGINAL_OBJECT')
        if not other.get('wgs84_bounds'): raise ValueError('HOLD_UNLOCATED_OPPOSITE_SOURCE')
        if overlaps(row['wgs84_bounds'], other['wgs84_bounds']): raise ValueError('HOLD_CROSS_SPLIT_FOOTPRINT')
        lower = geod.inv(*row['center'], *other['center'])[2] - radius(row) - radius(other)
        distances.append(lower)
    nearest = min(distances)
    if nearest < 1000: raise ValueError('HOLD_NEAR_OPPOSITE_SOURCE_WITHIN_1KM')
    return dict(nearest_opposite_footprint_lower_bound_m=nearest, proximity_hold_m=1000,
                scope='all cached opposite-role footprints, including holdout metadata; no holdout pixels')


def rights(row, inventory_cache):
    evidence = row['source_evidence']
    path = inventory_cache/(row['oam_id']+'-stac.json')
    require_hash(path, evidence['snapshot_sha256'])
    item = json.loads(path.read_text(encoding='utf-8-sig'))
    props = item['properties']
    providers = props.get('providers', [])
    if row['license'] != 'CC-BY 4.0' or item['id'] != row['oam_id'] or props.get('license', '').lower() != 'cc-by-4.0':
        raise ValueError('HOLD_SOURCE_LICENSE_OR_ID')
    if not any(p.get('name', '').strip() and 'licensor' in p.get('roles', []) for p in providers):
        raise ValueError('HOLD_ATTRIBUTION_INCOMPLETE')
    if not props.get('oam:producer_name', '').strip(): raise ValueError('HOLD_PRODUCER_MISSING')
    # The migrated STAC uses the OAM -temp bucket; require the same original metadata object key.
    if urlparse(item['assets']['metadata']['href']).path != urlparse(row['meta']).path:
        raise ValueError('HOLD_ORIGINAL_METADATA_KEY_MISMATCH')
    return item, dict(stac_sha256=sha(path), stac_url=evidence['url'], row_license=row['license'],
        image_license=props['license'], annotation_license='CC BY 4.0',
        annotation_terms=f'https://github.com/Restor-Foundation/tcd/blob/{CODE_REVISION}/docs/dataset.md',
        attribution='named producer/licensor retained in external STAC; no contacts published',
        commercial_final_clearance='NOT_ASSESSED')


class Network:
    """Persistent accounting; no retries; exact, identity-encoded HTTP byte ranges."""
    def __init__(self, cache, transfer_limit, storage_limit):
        self.cache, self.storage_limit = cache, storage_limit
        self.path = cache/'network.json'
        self.log = json.loads(self.path.read_text()) if self.path.exists() else dict(bytes=0, requests=[])
        self.limit = transfer_limit
        self.session = requests.Session()
        self.resolved = {}

    def get(self, url, start=None, size=None):
        if size is not None and size > self.limit-self.log['bytes']: raise ValueError('Transfer budget would be exceeded')
        headers = {'Accept-Encoding': 'identity'}
        if start is not None: headers['Range'] = f'bytes={start}-{start+size-1}'
        entry = dict(url=url, start=start, requested_bytes=size, received_bytes=0)
        self.log['requests'].append(entry)
        try:
            with self.session.get(self.resolved.get(url, url), headers=headers, stream=True, timeout=60) as response:
                entry['http_status'] = response.status_code
                response.raise_for_status()
                if response.headers.get('Content-Encoding', 'identity') != 'identity': raise ValueError('Unexpected content encoding')
                if start is not None:
                    expected = f'bytes {start}-{start+size-1}/'
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise ValueError('Server did not honor exact HTTP Range')
                    if int(response.headers.get('Content-Length', '-1')) != size: raise ValueError('Range content length mismatch')
                self.resolved[url] = response.url
                data = bytearray()
                while size is None or len(data) < size:
                    remaining = self.limit-self.log['bytes']
                    if remaining <= 0: raise ValueError('Transfer budget exhausted')
                    chunk = response.raw.read(min(65536, remaining, size-len(data) if size else 65536))
                    if not chunk: break
                    self.log['bytes'] += len(chunk); entry['received_bytes'] += len(chunk); data.extend(chunk)
                if size is not None and len(data) != size: raise ValueError('Incomplete byte range')
                entry['sha256'] = hashlib.sha256(data).hexdigest()
                return bytes(data)
        finally:
            dump(self.path, self.log)

    def cached(self, url, start, size):
        key = hashlib.sha256(f'{url}:{start}:{size}'.encode()).hexdigest()
        path = self.cache/'ranges'/(key+'.bin')
        index = path.with_suffix('.json')
        if path.exists():
            saved = json.loads(index.read_text()); require_hash(path, saved['sha256'])
            return path.read_bytes()
        storage_gate(self.cache, None, size+1024, self.storage_limit)
        data = self.get(url, start, size)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data); dump(index, dict(url=url, start=start, size=size, sha256=sha(path)))
        return data


class Chunks(io.RawIOBase):
    """Expose only prefetched column chunks; never fetch other columns implicitly."""
    def __init__(self, size, chunks): self.size, self.chunks, self.position = size, chunks, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0: raise ValueError('Negative seek')
        return self.position
    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position+size)
        if end <= self.position: return b''
        result = bytearray()
        while self.position < end:
            match = next(((start, data) for start, data in self.chunks
                          if start <= self.position < start+len(data)), None)
            if match is None: raise ValueError('Parquet requested unplanned bytes')
            start, data = match
            count = min(end-self.position, start+len(data)-self.position)
            result.extend(data[self.position-start:self.position-start+count]); self.position += count
        return bytes(result)


def group_for(metadata, index):
    offset = 0
    for group in range(metadata.num_row_groups):
        count = metadata.row_group(group).num_rows
        if offset <= index < offset+count: return group, index-offset
        offset += count
    raise ValueError('Shard row out of bounds')


def column_ranges(metadata, group):
    result = []
    rg = metadata.row_group(group)
    for index in range(rg.num_columns):
        column = rg.column(index)
        if column.path_in_schema.split('.')[0] not in COLUMNS: continue
        offsets = [v for v in [column.dictionary_page_offset, column.data_page_offset] if v is not None and v >= 0]
        result.append(dict(column=column.path_in_schema, start=min(offsets), size=column.total_compressed_size))
    return result


def bounded_selection(eligible, groups, remaining_transfer, remaining_storage):
    accepted, held = [], {}
    for row in sorted(eligible, key=lambda r: (r['image_id'] not in (2,9), r['image_id'])):
        tentative = accepted + [row['image_id']]
        used = [g for g in groups if any(t['image_id'] in tentative for t in g['targets'])]
        transfer = sum(g['compressed_bytes'] for g in used)
        raw_allowance = sum(r['height']*r['width']*6 + MIB for r in eligible if r['image_id'] in tentative)
        peak = max((g['compressed_bytes'] for g in used), default=0) + raw_allowance
        reason = ('HOLD_TRANSFER_ESTIMATE' if transfer > remaining_transfer else
                  'HOLD_STORAGE_ESTIMATE' if peak > remaining_storage else None)
        if reason: held[row['image_id']] = reason
        else: accepted = tentative
    return accepted, held


def plan(args, report):
    selected, rows, reference = load_inventory(args.inventory_cache, args.ids)
    report['candidates'] = []
    eligible = []
    for row in selected:
        entry = dict(image_id=row['image_id'], split=role(row), status='HOLD')
        report['candidates'].append(entry)
        try:
            entry['separation'] = separation(row, rows)
            _, entry['rights'] = rights(row, args.inventory_cache)
            eligible.append(row); entry['status'] = 'METADATA_RIGHTS_PASS_PIXELS_PENDING'
        except (ValueError, KeyError, FileNotFoundError) as error: entry['reason'] = str(error)
    network = Network(args.cache, args.transfer_mib*MIB, args.storage_mib*MIB)
    api_path = args.cache/'publisher.json'
    if not api_path.exists():
        raw = network.get(f'https://huggingface.co/api/datasets/restor/tcd/revision/{REVISION}?blobs=true')
        storage_gate(args.cache, None, len(raw), args.storage_mib*MIB); api_path.write_bytes(raw)
    publisher = json.loads(api_path.read_text())
    if publisher['sha'] != REVISION: raise ValueError('Publisher revision mismatch')
    shards = {s['rfilename']: s for s in publisher['siblings']}
    reference_shards = {s['path']: s for s in reference['acquisition']['shards']}
    report['groups'] = []
    for shard_name in sorted({r['shard'] for r in eligible}):
        if not shard_name.startswith('data/train-'): raise ValueError('Holdout shard forbidden')
        shard = shards[shard_name]
        if shard['lfs']['sha256'] != reference_shards[shard_name]['publisher_sha256']: raise ValueError('Publisher shard mismatch')
        url, size = ROOT+'/'+shard_name, shard['size']
        tail = network.cached(url, size-8, 8)
        if tail[4:] != b'PAR1': raise ValueError('Not a Parquet footer')
        footer_size = struct.unpack('<I', tail[:4])[0]
        if footer_size > MIB: raise ValueError('Unexpectedly large footer')
        footer = network.cached(url, size-8-footer_size, footer_size)
        metadata = pq.read_metadata(io.BytesIO(b'PAR1'+footer+tail))
        groups = {}
        for row in [r for r in eligible if r['shard'] == shard_name]:
            group, index = group_for(metadata, row['shard_row'])
            groups.setdefault(group, []).append(dict(image_id=row['image_id'], index=index))
        for group, targets in groups.items():
            ranges = column_ranges(metadata, group)
            report['groups'].append(dict(shard=shard_name, shard_size=size, group=group,
                group_rows=metadata.row_group(group).num_rows, targets=targets, ranges=ranges,
                compressed_bytes=sum(r['size'] for r in ranges), footer_size=footer_size))
    report.update(transfer_limit_bytes=args.transfer_mib*MIB, storage_limit_bytes=args.storage_mib*MIB,
                  inventory_sha256=reference['metadata_cache_sha256'], ids=args.ids)
    report['all_requested_column_bytes'] = sum(g['compressed_bytes'] for g in report['groups'])
    # Prefer one Train and one Val before a second Train. Evict each decoded group's
    # new compressed cache, preserving footer caches and request/hash provenance.
    accepted, held = bounded_selection(eligible, report['groups'], network.limit-network.log['bytes'],
                                      args.storage_mib*MIB-256*1024-stored(args.cache))
    for rid, reason in held.items():
        next(e for e in report['candidates'] if e['image_id'] == rid).update(status='HOLD',reason=reason)
    for group in report['groups']:
        group['acquire_targets'] = [t for t in group['targets'] if t['image_id'] in accepted]
    report['accepted_ids'] = accepted
    required = sum(g['compressed_bytes'] for g in report['groups'] if g['acquire_targets'])
    report['planned_column_bytes'] = required
    report['network_bytes'] = network.log['bytes']
    report['status'] = 'PLAN_READY' if accepted else 'HOLD_BUDGET_OR_EVIDENCE'
    report['cache_policy'] = 'prefetch one row group; extract selected rows; evict its column cache; retain footers and hashes'
    dump(args.cache/'plan.json', report)
    return report


def acquire(args, report):
    saved = json.loads((args.cache/'plan.json').read_text())
    if saved['revision'] != REVISION or saved['ids'] != args.ids or saved['status'] != 'PLAN_READY':
        raise ValueError('A matching successful plan is required')
    if args.transfer_mib*MIB != saved['transfer_limit_bytes'] or args.storage_mib*MIB != saved['storage_limit_bytes']:
        raise ValueError('Budget configuration differs from plan')
    selected, rows, _ = load_inventory(args.inventory_cache, args.ids)
    lookup = {r['image_id']: r for r in selected}
    accepted = saved['accepted_ids']
    if not accepted or not set(accepted) <= set(lookup): raise ValueError('Unapproved acquisition IDs')
    publisher = json.loads((args.cache/'publisher.json').read_text())
    if publisher['sha'] != REVISION: raise ValueError('Publisher revision mismatch')
    publishers = {s['rfilename']:s for s in publisher['siblings']}
    targets = [t['image_id'] for g in saved['groups'] for t in g['acquire_targets']]
    if len(targets) != len(set(targets)) or set(targets) != set(accepted): raise ValueError('Incomplete/duplicate acquisition targets')
    for group in saved['groups']:
        if not group['shard'].startswith('data/train-'): raise ValueError('Holdout shard forbidden')
        if group['shard_size'] != publishers[group['shard']]['size']: raise ValueError('Shard size conflict')
        for target in group['acquire_targets']:
            row = lookup[target['image_id']]
            if row['shard'] != group['shard']: raise ValueError('Wrong target shard')
            separation(row,rows); rights(row,args.inventory_cache)
    raw_root = args.cache/'raw'
    raw_root.mkdir(exist_ok=False)
    network = Network(args.cache, args.transfer_mib*MIB, args.storage_mib*MIB)
    report.update(plan_sha256=sha(args.cache/'plan.json'), samples=[], candidates=saved['candidates'])
    for group in saved['groups']:
        if not group['acquire_targets']: continue
        url = ROOT+'/'+group['shard']; size = group['shard_size']
        tail = network.cached(url, size-8, 8)
        footer = network.cached(url, size-8-group['footer_size'], group['footer_size'])
        metadata = pq.read_metadata(io.BytesIO(b'PAR1'+footer+tail))
        for target in group['acquire_targets']:
            expected = lookup[target['image_id']]
            if group_for(metadata,expected['shard_row']) != (group['group'],target['index']):
                raise ValueError('Planned row index mismatch')
        # Recompute the plan from the pinned footer instead of trusting offsets in editable JSON.
        if column_ranges(metadata, group['group']) != group['ranges']: raise ValueError('Column plan mismatch')
        chunks, paths = [], []
        for piece in group['ranges']:
            raw = network.cached(url, piece['start'], piece['size'])
            chunks.append((piece['start'], raw))
            key = hashlib.sha256(f"{url}:{piece['start']}:{piece['size']}".encode()).hexdigest()
            paths.append(args.cache/'ranges'/(key+'.bin'))
        remote = Chunks(size, chunks)
        parquet = pq.ParquetFile(remote, metadata=metadata, pre_buffer=False)
        table = parquet.read_row_group(group['group'], columns=COLUMNS, use_threads=False)
        for target in group['acquire_targets']:
            row = table.slice(target['index'], 1).to_pylist()[0]
            expected = lookup[target['image_id']]
            if any(row[k] != expected[k] for k in ['image_id','oam_id','license','validation_fold','height','width','crs','bounds','meta']):
                raise ValueError('Decoded shard row identity/metadata mismatch')
            if group_for(metadata, expected['shard_row']) != (group['group'], target['index']):
                raise ValueError('Row index mismatch')
            _, evidence = rights(expected, args.inventory_cache)
            leak = separation(expected, rows)
            row['split'] = role(expected)
            image_bytes, annotation_bytes = row['image']['bytes'], row['annotation']['bytes']
            storage_gate(args.cache, None, len(image_bytes)+len(annotation_bytes)+
                len(row['segments'].encode())+len(row['coco_annotations'].encode())+row['height']*row['width']+MIB,
                args.storage_mib*MIB)
            folder = raw_root/str(row['image_id'])
            hashes, qa = save_pair(folder, row, image_bytes, annotation_bytes)
            stac_data = (args.inventory_cache/(row['oam_id']+'-stac.json')).read_bytes()
            (folder/'stac.json').write_bytes(stac_data); hashes['stac.json'] = sha(folder/'stac.json')
            rgb = np.array(Image.open(folder/'rgb.bin'))
            mask = np.array(Image.open(folder/'candidate-mask.png'))
            preview = Image.new('RGB', (rgb.shape[1]*2, rgb.shape[0]))
            preview.paste(Image.fromarray(rgb)); preview.paste(Image.fromarray(overlay(rgb, mask)), (rgb.shape[1], 0))
            preview.thumbnail((1400, 700)); preview.save(folder/'preview.jpg', quality=90)
            hashes['preview.jpg'] = sha(folder/'preview.jpg')
            report['samples'].append(dict(metadata=expected, hashes=hashes, label_qa=qa,
                rights=evidence, separation=leak, visual_review='NOT_VERIFIED'))
        del table, parquet, remote, chunks, raw
        # Only newly created, explicitly named column-cache files are evicted. No recursive deletion.
        for path in paths:
            path.unlink(); path.with_suffix('.json').unlink()
        report['network_bytes'] = network.log['bytes']
        dump(args.cache/'raw_manifest.json', report)
    report.update(status='RAW_ACQUIRED_QA_PENDING', network_bytes=network.log['bytes'])
    dump(args.cache/'raw_manifest.json', report)
    return report


def inspect_source(folder, sample):
    row = sample['metadata']
    for name, expected in sample['hashes'].items(): require_hash(folder/name, expected)
    stac = json.loads((folder/'stac.json').read_text(encoding='utf-8-sig'))
    if stac['id'] != row['oam_id'] or stac['properties'].get('license','').lower() != 'cc-by-4.0':
        raise ValueError('STAC mismatch')
    rgb_image, ann_image = Image.open(folder/'rgb.bin'), Image.open(folder/'annotation.bin')
    if rgb_image.format != 'TIFF' or ann_image.format != 'PNG': raise ValueError('Unexpected source formats')
    if any(im.getexif().get(274, 1) != 1 for im in (rgb_image, ann_image)):
        raise ValueError('Unsupported orientation')
    rgb, annotation = np.array(rgb_image), np.array(ann_image)
    segments = json.loads((folder/'segments.json').read_text())
    mask, qa = labels(rgb, annotation, segments)
    if not np.array_equal(mask, np.array(Image.open(folder/'candidate-mask.png'))): raise ValueError('Mask regeneration differs')
    # Independently check known crowd uncompressed COCO RLE where present.
    ids = annotation.astype(np.uint32)
    ids = ids[:,:,0] + 256*ids[:,:,1] + 65536*ids[:,:,2]
    for obj in json.loads((folder/'coco_annotations.json').read_text()):
        if obj.get('category_id') not in (1,2) or obj.get('iscrowd') not in (0,1):
            raise ValueError('Unknown COCO category/crowd flag')
        if obj.get('iscrowd'):
            matches = [s for s in segments if s['iscrowd'] and s['category_id'] == obj['category_id']]
            if len(matches) != 1 or not np.array_equal(decode_rle(obj['segmentation']), ids == matches[0]['id']):
                raise ValueError('Crowd RLE mismatch/ambiguous')
    with rasterio.open(folder/'rgb.bin') as src:
        if str(src.crs) != 'EPSG:3395' or src.count != 3 or src.shape != (row['height'],row['width']):
            raise ValueError('Source CRS/dimensions mismatch')
        t = src.transform
        if t.b or t.d or t.a <= 0 or t.e >= 0: raise ValueError('Unsupported source affine')
        if src.nodata is not None or not (src.dataset_mask() == 255).all(): raise ValueError('Nodata/invalid needs review')
        if not np.array_equal(src.read().transpose(1,2,0), rgb): raise ValueError('Raster/PIL RGB mismatch')
        # HF stores bounds as float32: allow only its rounding uncertainty, not a spatial correction.
        tolerance = np.maximum(np.abs(np.spacing(np.array(row['bounds'], dtype=np.float32)))/2, 1e-8)
        if not np.all(np.abs(np.array(src.bounds)-row['bounds']) <= tolerance+1e-6):
            raise ValueError('Metadata/embedded footprint mismatch')
        geographic = rasterio.warp.transform_bounds(src.crs,4326,*src.bounds)
        b = stac['bbox']
        if not (b[0] <= geographic[0] <= geographic[2] <= b[2] and b[1] <= geographic[1] <= geographic[3] <= b[3]):
            raise ValueError('Tile outside original STAC footprint')
        shape, transform, effective = output_grid(src.crs, t, src.width, src.height)
        center = t*(src.width/2, src.height/2)
        info = dict(source_crs=str(src.crs), source_transform=list(t)[:6], source_bounds=list(src.bounds),
            source_wgs84_bounds=list(geographic), input_shape=list(src.shape),
            source_ground_gsd=intervals(src.crs,*center,t.a,-t.e), projected_pixel_m=list(src.res),
            output_crs=str(src.crs), output_transform=list(transform)[:6], output_bounds=list(src.bounds),
            output_shape=list(shape), output_ground_gsd=effective, target_ground_gsd_m=0.5,
            gsd_error_m={k:effective[k]-.5 for k in ['east_m','north_m']}, nodata=None, validity='all pixels reverified valid',
            pixel_definition='outer bounds unchanged; centers col+0.5,row+0.5',
            label_qa=qa, source_formats=['TIFF','PNG'])
    coverage = np.clip(reduce_area(mask,shape),0,1).astype(np.float32)
    rgb_out = np.floor(np.clip(reduce_area(rgb,shape),0,255)+.5).astype(np.uint8)
    valid = np.clip(reduce_area(np.ones(mask.shape,dtype=np.uint8),shape),0,1).astype(np.float32)
    products = {'rgb.tif':rgb_out.transpose(2,0,1), 'canopy_fraction.tif':coverage[None],
                'canopy_mask.tif':binary(coverage)[None], 'valid_fraction.tif':valid[None]}
    info.update(canopy_fraction_mean=float(coverage.mean()), binary_positive_pixels=int(binary(coverage).sum()),
        binary_canopy_ratio=float(binary(coverage).mean()), uniform_background=bool(np.all(mask==0)),
        uniform_canopy=bool(np.all(mask==1)), invalid_pixels=0, threshold=.5,
        rgb_method='exact pixel-overlap area float64; floor(value+0.5) uint8',
        fraction_method='same area weights float64 then float32; binary >=0.5',
        crowd_policy='known categories 1/2 positive; ID0 annotation background, not nodata')
    black = np.all(rgb == 0, axis=2)
    border = np.zeros(black.shape, dtype=bool)
    border[[0,-1],:] = True; border[:,[0,-1]] = True
    info.update(exact_black_pixels=int(black.sum()), black_border_pixels=int((black & border).sum()),
                gdal_all_valid=True)
    if np.any(black & border):
        # This is a review trigger, not a rule declaring every black pixel nodata.
        # Do not emit an all-one valid_fraction for an unresolved image footprint.
        info.update(source_validity_hold='HOLD_UNMARKED_BLACK_BORDER_VALIDITY',
                    invalid_pixels=None,
                    grid_status='diagnostic proposed grid only; no training files written',
                    coverage_scope='diagnostic full rectangle, unresolved footprint included',
                    validity='GDAL all-valid flag is insufficient; black edge requires source validity evidence')
        return {}, info
    return products, info


def review_gate(sample, reviews):
    review = reviews.get(str(sample['metadata']['image_id']), {})
    if any(review.get(k) != sample['hashes'][name] for k,name in
           [('preview_sha256','preview.jpg'),('stac_sha256','stac.json')]):
        return False
    return all(review.get(k) == 'PASS' for k in ['alignment','quality','attribution'])


def prepare(args, report):
    selected, rows, _ = load_inventory(args.inventory_cache, args.ids)
    raw_manifest = json.loads((args.cache/'raw_manifest.json').read_text())
    if raw_manifest['revision'] != REVISION or raw_manifest['status'] != 'RAW_ACQUIRED_QA_PENDING':
        raise ValueError('Incomplete raw acquisition')
    saved_plan = json.loads((args.cache/'plan.json').read_text())
    if args.storage_mib*MIB != saved_plan['storage_limit_bytes']:
        raise ValueError('Storage budget differs from plan')
    lookup = {r['image_id']: r for r in selected}
    fresh(args.output, [args.cache, args.inventory_cache])
    reviews = json.loads(args.reviews.read_text()) if args.reviews else {}
    report.update(samples=[], dataset='restor/tcd', target_ground_gsd_m=.5,
        raw_manifest_sha256=sha(args.cache/'raw_manifest.json'),
        versions={name:version(name) for name in ['numpy','Pillow','rasterio','pyproj','pyarrow']})
    for sample in raw_manifest['samples']:
        row = sample['metadata']
        entry = dict(image_id=row['image_id'], oam_id=row['oam_id'], split=role(row),
                     validation_fold=row['validation_fold'], status='HOLD', training_eligible=False)
        report['samples'].append(entry)
        try:
            if row != lookup[row['image_id']]: raise ValueError('Raw metadata differs from pinned inventory')
            _, rights_info = rights(row, args.inventory_cache)
            if sample['hashes']['stac.json'] != rights_info['stac_sha256']:
                raise ValueError('Raw STAC differs from pinned rights evidence')
            entry.update(rights=rights_info, separation=separation(row, rows), source_hashes=sample['hashes'])
            products, info = inspect_source(args.cache/'raw'/str(row['image_id']), sample)
            entry.update(info)
            if info.get('source_validity_hold'):
                entry.update(pixel_qa='PASS_IDS_DIMENSIONS', spatial_qa='PASS_GRID_VALIDITY_UNRESOLVED',
                             visual_qa='NOT_ACCEPTED', validity_qa='HOLD')
                raise ValueError(info['source_validity_hold'])
            storage_gate(args.cache, args.output, sum(a.nbytes for a in products.values())+MIB, args.storage_mib*MIB)
            folder = args.output/entry['split']/str(row['image_id']); folder.mkdir(parents=True)
            hashes = {}
            from affine import Affine
            for name, array in products.items():
                path = folder/name
                with rasterio.open(path,'w',driver='GTiff',height=array.shape[1],width=array.shape[2],count=array.shape[0],
                    dtype=array.dtype,crs=info['output_crs'],transform=Affine(*info['output_transform']),compress='deflate') as dst:
                    dst.write(array)
                hashes[name] = sha(path)
            entry.update(output_hashes=hashes, pixel_qa='PASS', spatial_qa='PASS',
                visual_qa='PASS' if review_gate(sample,reviews) else 'NOT_VERIFIED',
                attribution_reference=f"raw/{row['image_id']}/stac.json",
                visual_review=reviews.get(str(row['image_id']),{}))
            entry['training_eligible'] = entry['visual_qa'] == 'PASS'
            entry['status'] = 'PILOT_PAIR_PASS' if entry['training_eligible'] else 'PAIR_GENERATED_REVIEW_PENDING'
            verify_sample(args.output, entry, products)
        except (ValueError, KeyError) as error:
            entry.update(status='HOLD', training_eligible=False, reason=str(error))
    report['status'] = ('REAL_PILOT_PASS' if {s['split'] for s in report['samples'] if s['training_eligible']} == {'train','val'}
                        else 'REAL_PILOT_NOT_VERIFIED')
    report.update(network_bytes=json.loads((args.cache/'network.json').read_text())['bytes'],
                  total_stored_bytes_before_manifest=stored(args.cache,args.output))
    dump(args.output/'manifest.json', report)
    return report


def verify_sample(output, sample, expected_arrays=None):
    from affine import Affine
    if set(sample['output_hashes']) != PRODUCTS: raise ValueError('Incomplete product set')
    arrays = {}
    for name, expected in sample['output_hashes'].items():
        path = output/sample['split']/str(sample['image_id'])/name
        require_hash(path, expected)
        with rasterio.open(path) as src:
            if (str(src.crs) != sample['output_crs'] or src.transform != Affine(*sample['output_transform']) or
                list(src.shape) != sample['output_shape'] or not np.allclose(src.bounds,sample['output_bounds'],atol=1e-8,rtol=0)
                or src.nodata is not None or not (src.dataset_mask()==255).all()): raise ValueError('Product grid/validity mismatch')
            arrays[name] = src.read()
        if expected_arrays is not None and not np.array_equal(arrays[name], expected_arrays[name]):
            raise ValueError('Saved pixels differ from deterministic transformation')
    rgb, fraction, mask, valid = [arrays[n] for n in ['rgb.tif','canopy_fraction.tif','canopy_mask.tif','valid_fraction.tif']]
    if rgb.dtype != np.uint8 or rgb.shape[0] != 3 or mask.dtype != np.uint8 or mask.shape[0] != 1:
        raise ValueError('Product dtype/channels mismatch')
    for a in [fraction,valid]:
        if a.dtype != np.float32 or a.shape != mask.shape or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():
            raise ValueError('Invalid fraction')
    if not np.all(valid==1) or not np.array_equal(mask,binary(fraction)): raise ValueError('Mask/validity mismatch')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','acquire','prepare','verify'])
    p.add_argument('--inventory-cache',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--ids',type=int,nargs='+',default=[2,3,9])
    p.add_argument('--transfer-mib',type=int,default=256)
    p.add_argument('--storage-mib',type=int,default=128)
    p.add_argument('--reviews',type=Path,help='External human review keyed by ID and exact preview/STAC hashes')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not 1 <= args.transfer_mib <= 256 or not 1 <= args.storage_mib <= 128: p.error('Budget outside pilot limits')
    if args.stage in ['prepare','verify'] and args.output is None: p.error('--output required')
    report = dict(dataset='restor/tcd',revision=REVISION,official_code_revision=CODE_REVISION,
        utc=datetime.now(timezone.utc).isoformat(), status='INCOMPLETE', stage=args.stage)
    started = time.perf_counter()
    owned = False
    raw_existed = (args.cache/'raw').exists()
    try:
        if args.cache.resolve() == REPO or REPO in args.cache.resolve().parents:
            raise ValueError('External cache required')
        if args.stage == 'plan':
            # Check input cache before reserving a new destination. No automatic metadata regeneration.
            load_inventory(args.inventory_cache,args.ids)
            fresh(args.cache,[args.inventory_cache])
            owned = True
            plan(args,report)
        elif args.stage == 'acquire': acquire(args,report)
        elif args.stage == 'prepare': prepare(args,report)
        else:
            selected, rows, _ = load_inventory(args.inventory_cache,args.ids)
            lookup = {r['image_id']:r for r in selected}
            result = json.loads((args.output/'manifest.json').read_text())
            raw = json.loads((args.cache/'raw_manifest.json').read_text())
            require_hash(args.cache/'raw_manifest.json',result['raw_manifest_sha256'])
            verified, held = [], []
            for sample in result['samples']:
                original = next(s for s in raw['samples'] if s['metadata']['image_id']==sample['image_id'])
                row = original['metadata']
                if row != lookup[row['image_id']] or sample['split'] != role(row) or sample['oam_id'] != row['oam_id']:
                    raise ValueError('Sample provenance/split mismatch')
                separation(row,rows); _, evidence = rights(row,args.inventory_cache)
                if original['hashes']['stac.json'] != evidence['stac_sha256']: raise ValueError('STAC hash conflict')
                products, info = inspect_source(args.cache/'raw'/str(sample['image_id']),original)
                if info.get('source_validity_hold'):
                    if sample.get('reason') != info['source_validity_hold'] or sample['training_eligible']:
                        raise ValueError('Previously held source incorrectly accepted')
                    held.append(sample['image_id'])
                    continue
                if any(sample[k] != info[k] for k in ['output_shape','output_transform','output_bounds','output_crs']):
                    raise ValueError('Manifest grid differs from source-derived grid')
                verify_sample(args.output,sample,products); verified.append(sample['image_id'])
            if not verified and not held: raise ValueError('No samples to verify')
            report.update(status='OFFLINE_REVERIFICATION_COMPLETE_HOLDS' if held else 'OFFLINE_REVERIFICATION_PASS',
                          verified_product_ids=verified, held_source_ids=held)
    except Exception as error:
        report.update(status='FAIL_STOPPED', error=f'{type(error).__name__}: {error}')
    report['seconds'] = time.perf_counter()-started
    print(json.dumps(report,indent=2))
    # Stage artifacts retain structured diagnostics; verify is read-only and emits JSON to stdout.
    owned = owned or (args.stage == 'acquire' and not raw_existed and (args.cache/'raw').is_dir())
    if owned:
        path = args.cache/(args.stage+'-execution.json')
        if not path.exists(): dump(path, report)
    if report['status'] in ['REAL_PILOT_NOT_VERIFIED','OFFLINE_REVERIFICATION_COMPLETE_HOLDS']:
        return 3  # Operation completed, but data eligibility remains HOLD.
    return 0 if report['status'] in ['PLAN_READY','RAW_ACQUIRED_QA_PENDING','REAL_PILOT_PASS',
                                    'OFFLINE_REVERIFICATION_PASS'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
