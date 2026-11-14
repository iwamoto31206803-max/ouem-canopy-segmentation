"""Phase C1 direct grid-window crops for five exploratory research sources."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.transform import Affine, array_bounds

from m1_2d_geometry_audit import require_hash, sha
from m1_2d_prepare_pairs import output_grid, reduce_area, binary
from m1_5b2_border_audit import component_report
from m1_5b_pilot import rights
from m1_5d_zenodo_prepare import GENERATION, local_index, semantic_binary


TARGETS = {1054: 'train', 2493: 'train', 3206: 'train', 1241: 'val', 205: 'val'}
POLICY = 'r1c0-edge-black-positive-support-v1'


def window_sums(array, size):
    if min(array.shape) < size:
        return np.zeros((0, 0), dtype=np.int64)
    sums = np.pad(array.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return sums[size:, size:]-sums[:-size, size:]-sums[size:, :-size]+sums[:-size, :-size]


def blocked_grid(mask, shape):
    """Exact integer floor/ceil support from Phase B, including partial cells."""
    h, w = mask.shape
    oh, ow = shape
    y0 = np.arange(oh, dtype=np.int64)*h//oh
    y1 = ((np.arange(oh, dtype=np.int64)+1)*h+oh-1)//oh
    x0 = np.arange(ow, dtype=np.int64)*w//ow
    x1 = ((np.arange(ow, dtype=np.int64)+1)*w+ow-1)//ow
    sums = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (sums[y1[:, None], x1]-sums[y0[:, None], x1]
            -sums[y1[:, None], x0]+sums[y0[:, None], x0]) > 0


def choose_window(rgb, teacher, shape, image_id, seed=0, size=224):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(size) is not int or size <= 0:
        raise ValueError('Integer seed and positive patch size required')
    black, border_black, _ = component_report(np.all(rgb == 0, axis=2), teacher, 4)
    white, border_white, _ = component_report(np.all(rgb == 255, axis=2), teacher, 4)
    blocked = blocked_grid(border_black, shape)
    candidates = np.argwhere(window_sums(blocked, size) == 0)
    if not len(candidates):
        raise ValueError('No direct clean window; padding/upscale forbidden')
    # Stable per-source selection, independent of processing order or model scores.
    y, x = candidates[random.Random(f'{seed}:{image_id}').randrange(len(candidates))]
    y, x = int(y), int(x)
    white_cells = blocked_grid(border_white, shape)
    return dict(crop_xywh=[x, y, size, size], placement_count=len(candidates),
                bw_placement_count=int((window_sums(blocked | white_cells, size) == 0).sum()),
                black=black, white=white,
                selected_black_support_cells=int(blocked[y:y+size, x:x+size].sum()),
                selected_white_support_cells=int(white_cells[y:y+size, x:x+size].sum()))


def crop_arrays(rgb, teacher, shape, grid, crop):
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x+width > shape[1] or y+height > shape[0]:
        raise ValueError('Crop outside output grid')
    rgb_grid = np.floor(np.clip(reduce_area(rgb, shape), 0, 255)+.5).astype(np.uint8)
    fraction_grid = np.clip(reduce_area(teacher, shape), 0, 1).astype(np.float32)
    fraction = fraction_grid[y:y+height, x:x+width]
    arrays = {'rgb.tif': rgb_grid[y:y+height, x:x+width].transpose(2, 0, 1),
              'canopy_fraction.tif': fraction[None], 'canopy_mask.tif': binary(fraction)[None]}
    return arrays, grid*Affine.translation(x, y)


def write_products(folder, arrays, crs, transform):
    folder.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, array in arrays.items():
        path = folder/name
        profile = dict(driver='GTiff', height=array.shape[1], width=array.shape[2], count=array.shape[0],
                       dtype=array.dtype, crs=crs, transform=transform, compress='deflate')
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(array)
        with rasterio.open(path) as src:
            if (not np.array_equal(src.read(), array) or src.crs != rasterio.crs.CRS.from_user_input(crs)
                    or src.transform != transform or src.shape != array.shape[1:]
                    or src.dtypes != (str(array.dtype),)*array.shape[0] or not np.isfinite(array).all()
                    or not np.allclose(src.bounds, array_bounds(*src.shape, transform), rtol=0, atol=1e-8)):
                raise ValueError('Product readback/grid mismatch')
        hashes[name] = sha(path)
    with rasterio.open(folder/'canopy_fraction.tif') as src:
        saved_fraction = src.read()
    with rasterio.open(folder/'canopy_mask.tif') as src:
        if not np.array_equal(src.read(), (saved_fraction >= .5).astype(np.uint8)):
            raise ValueError('Saved float32 fraction/mask mismatch')
    return hashes


def prepare(root, diagnosis, diagnosis_sha256, evidence, output, seed=0, size=224):
    """Explicit research route. No R1a receipts or legacy real-mode invocation."""
    root, evidence, output = Path(root).resolve(), Path(evidence).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.exists() or any(p == output or p in output.parents or output in p.parents for p in (root, evidence, repo)):
        raise ValueError('Fresh output outside inputs/repository required')
    require_hash(Path(diagnosis), diagnosis_sha256)
    previous = json.loads(Path(diagnosis).read_text(encoding='utf-8'))
    prior = {r['image_id']: r for r in previous['images']}
    indexed = {r['image_id']: r for r in local_index(root)}
    selected = []
    pins = {}
    for rid, split in TARGETS.items():
        row = dict(indexed[rid])
        role = 'val' if row['validation_fold'] == 0 else 'train'
        if row['official_split'] != 'train' or role != split or prior[rid]['split'] != split:
            raise ValueError('Fixed source split mismatch')
        row['source_evidence'] = dict(snapshot_sha256=sha(evidence/(row['oam_id']+'-stac.json')),
            url='https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'+row['oam_id'])
        item, proof = rights(row, evidence)
        proof['producer'] = item['properties']['oam:producer_name']
        proof['attribution'] = (f"{proof['producer']}; OpenAerialMap item {row['oam_id']}; CC BY 4.0. "
            'OAM-TCD dataset/annotations, Restor, Zenodo 11617167 v1.0.0. Area resampled and cropped for research.')
        stem = f"{row['oam_id']}_{rid}"
        source_paths = dict(rgb=root/f'extracted/dataset/holdout/images/{stem}.tif',
                            annotation=root/f'extracted/dataset/holdout/masks/{stem}.png')
        for key, path in source_paths.items():
            require_hash(path, prior[rid]['input_sha256'][key])
            pins[path] = sha(path)
        selected.append((row, proof, source_paths))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='ouem-exploratory-crops-v1', purpose='PHASE_C1_RESEARCH_ONLY',
        status='FAIL', source_generation=GENERATION, policy=POLICY, seed=seed,
        selection='row-major candidates; random.Random(str(seed)+":"+str(image_id)).randrange(count)',
        diagnosis_sha256=diagnosis_sha256, r1_acceptance='UNCHANGED_HOLD', samples=[])
    result['metadata_sha256'] = {name: sha(root/'archives'/name) for name in ('train_meta.json','test_meta.json')}
    previews = []
    try:
        for row, proof, paths in selected:
            rid = row['image_id']
            with rasterio.open(paths['rgb']) as src:
                if (src.count != 3 or src.dtypes != ('uint8',)*3 or str(src.crs) != row['crs']
                        or src.shape != (row['height'], row['width']) or src.transform.b or src.transform.d
                        or src.transform.a <= 0 or src.transform.e >= 0
                        or not np.allclose(src.bounds, row['bounds'], rtol=0, atol=1e-8)):
                    raise ValueError('Source RGB format/geometry mismatch')
                if src.nodata is not None or not np.all(src.dataset_mask() == 255):
                    raise ValueError('Source nodata/mask requires review')
                rgb = src.read().transpose(1, 2, 0)
                shape, grid, gsd = output_grid(src.crs, src.transform, src.width, src.height)
                source_grid = dict(crs=str(src.crs), affine=list(src.transform)[:6], bounds=list(src.bounds))
            with Image.open(paths['annotation']) as im:
                if im.mode != 'RGB': raise ValueError('Annotation RGB required')
                annotation = np.asarray(im)
            teacher = semantic_binary(rgb, annotation, [1, 2])
            chosen = choose_window(rgb, teacher, shape, rid, seed, size)
            if (list(shape) != prior[rid]['geometry']['output_shape'] or chosen['placement_count'] !=
                    prior[rid]['scenarios']['edge_black_only']['patches'][str(size)]['placement_count']):
                raise ValueError('Phase B grid/candidate count mismatch')
            arrays, transform = crop_arrays(rgb, teacher, shape, grid, chosen['crop_xywh'])
            split = TARGETS[rid]
            hashes = write_products(output/split/str(rid), arrays, row['crs'], transform)
            entry = dict(image_id=rid, oam_id=row['oam_id'], split=split, validation_fold=row['validation_fold'],
                source_hashes={k: pins[p] for k, p in paths.items()}, source_metadata=row['meta'], rights=proof,
                source_grid=source_grid, full_grid=dict(shape=list(shape), affine=list(grid)[:6], ground_gsd=gsd),
                patch_grid=dict(shape=[size,size], crs=row['crs'], affine=list(transform)[:6],
                    bounds=list(array_bounds(size,size,transform))), **chosen,
                products={n: dict(path=f'{split}/{rid}/{n}',sha256=h) for n,h in hashes.items()},
                canopy_fraction_mean=float(arrays['canopy_fraction.tif'].mean()),
                canopy_mask_fraction=float(arrays['canopy_mask.tif'].mean()), product_readback='MATCHED',
                coverage_review='PENDING_VISUAL_REVIEW', per_segment_category_crosswalk='NOT_VERIFIED')
            result['samples'].append(entry)
            image = arrays['rgb.tif'].transpose(1,2,0)
            mask = arrays['canopy_mask.tif'][0].astype(bool)
            colored = np.zeros_like(image); colored[mask] = [0,210,0]
            overlay = image.copy(); overlay[mask] = ((image[mask].astype(np.uint16)+colored[mask])/2).astype(np.uint8)
            previews.append((rid, split, image, colored, overlay))
        for path, expected in pins.items(): require_hash(path, expected)
        sheet = Image.new('RGB', (size*3, (size+30)*len(previews)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (rid, split, image, mask, overlay) in enumerate(previews):
            draw.text((4, i*(size+30)+5), f'{rid} {split} | RGB / canopy=green / overlay', fill='black')
            for j, a in enumerate((image, mask, overlay)):
                sheet.paste(Image.fromarray(a), (j*size, i*(size+30)+30))
        sheet.save(output/'preview.png')
        result['preview_sha256'] = sha(output/'preview.png')
        result['status'] = 'EXPLORATORY_PILOT_PRODUCTS_COMPLETE'
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        with (output/'crop_plan.json').open('x',encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output/'crops.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image_id','split','crop_xywh','canopy_mask_fraction','white_support_cells'])
        for s in result['samples']: writer.writerow([s['image_id'],s['split'],s['crop_xywh'],s['canopy_mask_fraction'],s['selected_white_support_cells']])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','diagnosis','evidence','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--diagnosis-sha256',required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    report=prepare(args.root,args.diagnosis,args.diagnosis_sha256,args.evidence,args.output,args.seed)
    print(json.dumps(dict(status=report['status'],pairs=len(report['samples']))))
