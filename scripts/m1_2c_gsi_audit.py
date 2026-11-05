"""Bounded z18 GSI audit at three Komae candidate sites."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import requests

R = 6378137.0
Z = 18
SITES = [('A_cityhall', 35.63466, 139.57820),
         ('B_nogawa', 35.642955, 139.578622),
         ('C_nishigawara', 35.632894, 139.566725)]


def tile(lat, lon, z=Z):
    n = 2**z
    return (int((lon + 180) / 360 * n),
            int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))


def grid(x, y, z=Z):
    resolution = 2 * math.pi * R / (256 * 2**z)
    return resolution, -math.pi * R + x * 256 * resolution, math.pi * R - y * 256 * resolution


def ground(lat, lon, resolution):
    from pyproj import Geod, Transformer
    forward = Transformer.from_crs(4326, 3857, always_xy=True)
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)
    x, y = forward.transform(lon, lat)
    geod = Geod(ellps='WGS84')
    east = inverse.transform(x + resolution, y)
    north = inverse.transform(x, y + resolution)
    return {'east_m': geod.inv(lon, lat, *east)[2],
            'north_m': geod.inv(lon, lat, *north)[2],
            'spherical_approx_m': resolution * math.cos(math.radians(lat))}


def acquire(output):
    import rasterio
    from rasterio.transform import Affine
    repo = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if output.exists() or output == repo or repo in output.parents:
        raise ValueError('Use fresh external cache')
    output.mkdir(parents=True)
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'layer': 'seamlessphoto',
              'zoom': Z, 'credit': '国土地理院の地理院タイルを加工して作成',
              'source': 'https://maps.gsi.go.jp/development/ichiran.html#seamlessphoto',
              'sites': [], 'status': 'INCOMPLETE'}
    total = 0
    try:
        for name, lat, lon in SITES:
            folder = output / name
            folder.mkdir()
            cx, cy = tile(lat, lon)
            site = dict(name=name, latitude=lat, longitude=lon, tiles=[])
            record['sites'].append(site)
            mosaic = np.zeros((768, 768, 3), dtype=np.uint8)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    x, y = cx + dx, cy + dy
                    url = f'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{Z}/{x}/{y}.jpg'
                    with requests.get(url, stream=True, timeout=30) as response:
                        response.raise_for_status()
                        chunks, size = [], 0
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            total += len(chunk)
                            if size > 1024**2 or total > 32 * 1024**2:
                                raise ValueError('Tile/total transfer cap exceeded')
                            chunks.append(chunk)
                        raw = b''.join(chunks)
                        image = Image.open(io.BytesIO(raw))
                        if image.format != 'JPEG' or image.size != (256, 256):
                            raise ValueError('Unexpected tile format/dimensions')
                        array = np.array(image.convert('RGB'))
                        mosaic[(dy+1)*256:(dy+2)*256, (dx+1)*256:(dx+2)*256] = array
                        (folder / f'{Z}_{x}_{y}.jpg').write_bytes(raw)
                        site['tiles'].append(dict(z=Z, x=x, y=y, url=url, status=response.status_code,
                            content_type=response.headers.get('Content-Type'), bytes=len(raw), width=256, height=256,
                            format=image.format, sha256=hashlib.sha256(raw).hexdigest(), utc=datetime.now(timezone.utc).isoformat()))
            resolution, left, top = grid(cx-1, cy-1)
            transform = Affine(resolution, 0, left, 0, -resolution, top)
            path = folder / 'mosaic.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=768, height=768, count=3,
                               dtype='uint8', crs='EPSG:3857', transform=transform, compress='deflate') as dst:
                dst.write(mosaic.transpose(2, 0, 1))
                dst.update_tags(credit=record['credit'])
            Image.fromarray(mosaic).save(folder / 'mosaic.png')
            with rasterio.open(path) as src:
                if not np.array_equal(src.read().transpose(1, 2, 0), mosaic):
                    raise ValueError('GeoTIFF readback mismatch')
                site['geotiff'] = dict(crs=str(src.crs), transform=list(src.transform)[:6],
                    width=src.width, height=src.height, bounds=list(src.bounds),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pixel_readback_equal=True)
            site['map_pixel_m'] = resolution
            site['ground_pixel'] = ground(lat, lon, resolution)
        record['status'] = 'ACQUIRED_GRID_VERIFIED'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    record['download_bytes'] = total
    (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record['status'] == 'ACQUIRED_GRID_VERIFIED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(acquire(parser.parse_args().output))
