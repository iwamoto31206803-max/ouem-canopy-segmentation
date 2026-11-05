"""Offline bounds-preserving local-ground-scale preprocessing for two approved pairs."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from m1_2a_data_qa import labels
from m1_2d_geometry_audit import audit, intervals, sha


def area_weights(source, destination):
    if not 1 <= destination <= source:
        raise ValueError('Expected a reduction to positive dimensions')
    edges = np.linspace(0, source, destination + 1, dtype=np.float64)
    left = np.maximum(edges[:-1, None], np.arange(source)[None, :])
    right = np.minimum(edges[1:, None], np.arange(1, source+1)[None, :])
    return np.maximum(right-left, 0) / (source/destination)


def reduce_area(array, shape):
    wy = area_weights(array.shape[0], shape[0])
    wx = area_weights(array.shape[1], shape[1])
    if array.ndim == 2:
        return wy @ array.astype(np.float64) @ wx.T
    return np.stack([wy @ array[:, :, i].astype(np.float64) @ wx.T
                     for i in range(array.shape[2])], axis=-1)


def binary(fraction):
    return (fraction >= 0.5).astype(np.uint8)


def output_grid(crs, transform, width, height):
    x,y = transform * (width/2, height/2)
    source = intervals(crs,x,y,transform.a,-transform.e)
    out_width = max(1, int(np.floor(width*source['east_m']/0.5 + 0.5)))
    out_height = max(1, int(np.floor(height*source['north_m']/0.5 + 0.5)))
    left,top = transform * (0,0)
    right,bottom = transform * (width,height)
    grid = from_bounds(left,bottom,right,top,out_width,out_height)
    effective = intervals(crs,x,y,grid.a,-grid.e)
    return (out_height,out_width), grid, effective


def prepare(cache, output, manifest):
    source_audit = audit(cache,manifest)
    if output.exists() or Path(__file__).resolve().parents[1] in output.resolve().parents:
        raise ValueError('Use a fresh external output directory')
    for source in source_audit['samples']:
        if not source['gdal_all_valid'] or source['nodata'] is not None or not source['contained_in_stac_bbox']:
            raise ValueError('Source validity/footprint gate failed')
    output.mkdir(parents=True)
    result = dict(dataset=manifest['dataset'],revision=manifest['revision'],target_ground_gsd_m=0.5,
                  status='PASS_CANDIDATE',versions={p:version(p) for p in ['numpy','Pillow','rasterio','pyproj']},samples=[])
    for sample, source in zip(manifest['samples'],source_audit['samples']):
        folder=cache/str(sample['image_id'])
        rgb=np.array(Image.open(folder/'rgb.bin'))
        annotation=np.array(Image.open(folder/'annotation.bin'))
        mask,qa=labels(rgb,annotation,json.loads((folder/'segments.json').read_text()))
        with rasterio.open(folder/'rgb.bin') as src:
            if str(src.crs) != 'EPSG:3395' or src.transform.a <= 0 or src.transform.e >= 0:
                raise ValueError('Unexpected CRS/grid orientation')
            shape,transform,effective=output_grid(src.crs,src.transform,src.width,src.height)
            rgb_out=np.floor(np.clip(reduce_area(rgb,shape),0,255)+0.5).astype(np.uint8)
            coverage=np.clip(reduce_area(mask,shape),0,1).astype(np.float32)
            valid=np.clip(reduce_area(np.ones(mask.shape,dtype=np.uint8),shape),0,1).astype(np.float32)
            products={'rgb.tif':rgb_out.transpose(2,0,1), 'canopy_fraction.tif':coverage[None],
                      'canopy_mask.tif':binary(coverage)[None], 'valid_fraction.tif':valid[None]}
            destination=output/sample['split']/str(sample['image_id'])
            destination.mkdir(parents=True)
            hashes={}
            for name,array in products.items():
                path=destination/name
                with rasterio.open(path,'w',driver='GTiff',height=shape[0],width=shape[1],count=array.shape[0],
                                   dtype=array.dtype,crs=src.crs,transform=transform,compress='deflate') as dst:
                    dst.write(array)
                with rasterio.open(path) as saved:
                    if saved.crs != src.crs or saved.transform != transform or saved.shape != shape:
                        raise ValueError('Saved grid mismatch')
                    if not np.allclose(saved.bounds,src.bounds,atol=1e-8,rtol=0):
                        raise ValueError('Saved geographic extent mismatch')
                    if not np.array_equal(saved.read(),array):
                        raise ValueError('Saved pixel readback mismatch')
                hashes[name]=sha(path)
            for historical_key in ['proposed_crop','proposed_output_shape','fivefold_ground_pixel_m','nominal_gsd_m']:
                source.pop(historical_key, None)
            source.update(status='PASS_CANDIDATE',training_pair_generated=True,output_shape=list(shape),
                          output_crs=str(src.crs),output_transform=list(transform)[:6],output_bounds=list(src.bounds),
                          crop=[0,0,src.width,src.height],output_ground_gsd=effective,
                          error_from_target_m={k:effective[k]-0.5 for k in ['east_m','north_m']},
                          output_hashes=hashes,label_qa=qa,threshold=0.5,validity='all-valid assumption reverified',
                          rgb_method='separable exact pixel-overlap box area, float64 sum; floor(value+0.5) uint8',
                          fraction_method='same overlap weights, float64 sum then float32 save',
                          row_license=sample['license'], source_license='CC-BY-4.0', annotation_license='CC BY 4.0',
                          audit_reference='M1_2B_RIGHTS_SEMANTICS.md',
                          eligibility='user accepted assigned split; two samples only',
                          boundary_definition='source outer edges preserved; centers at col+0.5,row+0.5')
            result['samples'].append(source)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(Path('docs/implementation/M1_2A_SAMPLE_MANIFEST.json').read_text())
    print(json.dumps(prepare(args.cache,args.output,manifest),indent=2))
