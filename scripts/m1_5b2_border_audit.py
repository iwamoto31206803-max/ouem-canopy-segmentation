"""Offline black-border investigation; diagnostic arrays are never training validity masks."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from m1_2a_data_qa import labels
from m1_2b_overlay_qa import overlay
from m1_2d_geometry_audit import require_hash, sha, intervals
from m1_5b_pilot import fresh

REPO = Path(__file__).resolve().parents[1]


def components(mask, connectivity=4):
    """Row-run union/find, avoiding a new scipy/OpenCV dependency. Ends are exclusive."""
    if mask.ndim != 2 or mask.dtype != bool or connectivity not in (4,8):
        raise ValueError('Expected a boolean 2D array and connectivity 4/8')
    output = np.zeros(mask.shape,dtype=np.int32)
    parents = [0]
    previous = []
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for y,line in enumerate(mask):
        changes = np.diff(np.r_[False,line,False].astype(np.int8))
        starts,ends = np.flatnonzero(changes==1),np.flatnonzero(changes==-1)
        current = []; p = 0
        for start,end in zip(starts.tolist(),ends.tolist()):
            number = len(parents); parents.append(number)
            output[y,start:end] = number; current.append((start,end,number))
            while p < len(previous) and previous[p][1] < start+(connectivity==4): p += 1
            q = p
            while q < len(previous) and previous[q][0] < end+(connectivity==8):
                a,b = root(number),root(previous[q][2])
                parents[max(a,b)] = min(a,b)
                q += 1
        previous = current
    lookup = np.array([root(i) for i in range(len(parents))],dtype=np.int32)
    return lookup[output]


def component_report(mask, target, connectivity=4):
    ids = components(mask,connectivity)
    counts = np.bincount(ids.ravel())
    present = np.flatnonzero(counts[1:])+1
    edge = np.unique(np.r_[ids[0],ids[-1],ids[:,0],ids[:,-1]])
    edge = edge[edge!=0]
    border = np.isin(ids,edge)
    largest = []
    for number in sorted(present,key=lambda i:(-int(counts[i]),int(i)))[:5]:
        yy,xx = np.where(ids==number)
        largest.append(dict(area=int(counts[number]),image_fraction=float(counts[number]/mask.size),
            bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            touches_border=bool(number in edge),positive_label_pixels=int(target[ids==number].sum()),
            bbox_occupancy=float(counts[number]/((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)))))
    internal = mask & ~border
    return dict(connectivity=connectivity,components=len(present),black_pixels=int(mask.sum()),
        border_components=len(edge),border_connected_pixels=int(border.sum()),
        interior_components=int(len(present)-len(edge)),interior_pixels=int(internal.sum()),
        border_positive_label_pixels=int(target[border].sum()),interior_positive_label_pixels=int(target[internal].sum()),
        largest=largest),border,internal


def largest_clear_square(blocked):
    """Geometric capacity hypothesis only; does not certify clear pixels valid."""
    previous = np.zeros(blocked.shape[1]+1,dtype=np.int32)
    best = (0,0,0)
    for y,line in enumerate(blocked):
        current = np.zeros_like(previous)
        for x in range(blocked.shape[1]):
            if not line[x]:
                current[x+1] = 1+min(previous[x],previous[x+1],current[x])
                if current[x+1] > best[0]: best = (int(current[x+1]),x+1,y+1)
        previous = current
    size,x,y = best
    return dict(side_px=size,bbox_xyxy=[x-size,y-size,x,y],
                meaning='avoids only hypothesized border-connected dark component; not a certified valid crop')


def boundary_report(rgb,black):
    outside = np.zeros(black.shape,dtype=bool)
    outside[1:] |= black[:-1]; outside[:-1] |= black[1:]
    outside[:,1:] |= black[:,:-1]; outside[:,:-1] |= black[:,1:]
    outside &= ~black
    values = rgb[outside].max(axis=1)
    return dict(black_nonblack_4edge_count=int(np.count_nonzero(black[1:] != black[:-1])+
                                              np.count_nonzero(black[:,1:] != black[:,:-1])),
        adjacent_nonblack_pixels=int(outside.sum()),
        adjacent_nonblack_maxchannel_percentiles=np.percentile(values,[0,25,50,75,100]).tolist() if values.size else [],
        interpretation='pixel-edge/near-black statistics; JPEG ringing and dark real surfaces are not distinguished automatically')


def analyze(rgb,target):
    if rgb.shape != (*target.shape,3) or rgb.dtype != np.uint8:
        raise ValueError('RGB/teacher shape or dtype mismatch')
    black = np.all(rgb==0,axis=2)
    exact,border,interior = component_report(black,target,4)
    diagonal,_,_ = component_report(black,target,8)
    sensitivity = []
    near_border = None
    for threshold in [5,10]:
        candidate = np.all(rgb<=threshold,axis=2)
        stats,dark_edge,_ = component_report(candidate,target,4)
        sensitivity.append(dict(threshold=threshold,**stats))
        if threshold==5: near_border = dark_edge
    return dict(exact_4=exact,exact_8=diagonal,near_black_sensitivity=sensitivity,
                boundary=boundary_report(rgb,black),
                hypothetical_clear_square=largest_clear_square(near_border),
                nonblack_fraction=float((~black).mean()),
                nonblack_meaning='not a measured valid fraction'),border,interior,near_border


def audit(cache,output):
    reference = json.loads((REPO/'docs/implementation/M1_5B_PILOT_RESULT.json').read_text())
    require_hash(cache/'raw_manifest.json',reference['raw_manifest_sha256'])
    raw = json.loads((cache/'raw_manifest.json').read_text())
    if {s['metadata']['image_id'] for s in raw['samples']} != {2,9}:
        raise ValueError('Only existing pilot 2/9 is supported')
    fresh(output,[cache])
    result = dict(scope='Offline investigation only; no new images, training pairs, validity rules or training',
                  dataset_revision=reference['revision'],
                  versions={p:version(p) for p in ['numpy','Pillow','rasterio']},samples=[])
    for sample in raw['samples']:
        rid = sample['metadata']['image_id']; folder = cache/'raw'/str(rid)
        accepted = next(s for s in reference['samples'] if s['image_id']==rid)
        if sample['hashes'] != accepted['source_hashes']: raise ValueError('Pilot source provenance mismatch')
        for name,expected in sample['hashes'].items(): require_hash(folder/name,expected)
        im = Image.open(folder/'rgb.bin'); rgb = np.array(im)
        ann = np.array(Image.open(folder/'annotation.bin'))
        target,qa = labels(rgb,ann,json.loads((folder/'segments.json').read_text()))
        started = time.perf_counter()
        stats,border,interior,near = analyze(rgb,target)
        entry = dict(image_id=rid,source_hashes=sample['hashes'],label_qa=qa,**stats)
        stac = json.loads((folder/'stac.json').read_text(encoding='utf-8-sig'))
        with rasterio.open(folder/'rgb.bin') as src:
            tags = im.tag_v2
            entry['tiff'] = dict(shape=list(src.shape),bands=src.count,dtypes=list(src.dtypes),
                nodata=src.nodata,mask_flags=[[v.name for v in flags] for flags in src.mask_flag_enums],
                dataset_mask_values=np.unique(src.dataset_mask()).tolist(),
                band_mask_values=[np.unique(m).tolist() for m in src.read_masks()],
                color_interpretation=[v.name for v in src.colorinterp],alpha_present=any(v.name=='alpha' for v in src.colorinterp),
                sidecar_mask_present=any(p.endswith('.msk') for p in src.files),
                internal_mask_exposed_by_gdal=any(v.name=='per_dataset' for flags in src.mask_flag_enums for v in flags),
                tiff_ifd_frames=im.n_frames,tag_ids=sorted(tags.keys()),
                selected_tags={str(k):str(tags[k]) if k in tags else None for k in [259,262,274,338,42112,42113]},
                description_tag_present=270 in tags,software_tag_present=305 in tags,
                crs=str(src.crs),transform=list(src.transform)[:6],bounds=list(src.bounds),
                area_or_point=src.tags().get('AREA_OR_POINT'),gdal_tag_keys=list(src.tags()),
                compression=src.profile.get('compress'),block_shapes=src.block_shapes)
            x,y = src.transform*(src.width/2,src.height/2)
            gsd = intervals(src.crs,x,y,*src.res)
            side = stats['hypothetical_clear_square']['side_px']
            entry['hypothetical_clear_square'].update(ground_side_m=[side*gsd[k] for k in ['east_m','north_m']],
                approximate_pixels_at_050m=[int(np.floor(side*gsd[k]/.5+.5)) for k in ['east_m','north_m']],
                sufficiency='geometric only, not eligibility or verified coverage')
            geometry = transform_geom('EPSG:4326',src.crs,stac['geometry'])
            nominal_inside = geometry_mask([geometry],out_shape=src.shape,transform=src.transform,invert=True)
            entry['stac_geometry_pixel_centers'] = dict(inside_pixels=int(nominal_inside.sum()),
                outside_pixels=int((~nominal_inside).sum()),black_inside=int((nominal_inside & np.all(rgb==0,axis=2)).sum()),
                black_outside=int((~nominal_inside & np.all(rgb==0,axis=2)).sum()),
                meaning='nominal STAC source extent only, not ground-truth image validity')
        visual = stac['assets']['visual']
        projected_ring = np.array(transform_geom(visual['proj:code'],'EPSG:4326',
                                                visual['proj:geometry'])['coordinates'][0])
        item_ring = np.array(stac['geometry']['coordinates'][0][0])
        corner_error = float(max(min(np.max(np.abs(projected_ring-point),axis=1)) for point in item_ring))
        entry['stac'] = dict(id=stac['id'],bbox=stac['bbox'],geometry=stac['geometry'],
            property_keys=sorted(stac['properties']),extensions=stac.get('stac_extensions',[]),
            asset_keys=sorted(stac['assets']),source_raster_metadata={k:v for k,v in visual.items() if k.startswith('proj:') or k in ['type','file:size','roles']},
            raster_band_nodata_declared=any(k.startswith('raster:') for k in visual),
            explicit_validity_asset_present=any(k.lower() in ['mask','alpha','validity','footprint'] for k in stac['assets']),
            item_geometry_vs_projected_rectangle_max_corner_error_degrees=corner_error,
            upstream_source_tiff_content_inspected=False)
        diagnostic = rgb.copy(); diagnostic[border]=[255,0,255]; diagnostic[interior]=[0,255,255]
        canvas = Image.new('RGB',(rgb.shape[1]*3,rgb.shape[0]))
        for column,array in enumerate([rgb,diagnostic,overlay(rgb,target)]):
            canvas.paste(Image.fromarray(array),(rgb.shape[1]*column,0))
        canvas.thumbnail((1800,600)); path=output/f'{rid}-distribution.jpg'; canvas.save(path,quality=92)
        # Context samples show a transition from the largest border component and internal dark pixels.
        for name,mask in [('edge',border),('interior',interior),('interior-positive',interior & (target==1))]:
            positions = np.argwhere(mask)
            if name=='edge':
                transition = np.argwhere(mask[:,:-1] & ~mask[:,1:])
                if len(transition): positions=transition
            if name=='interior':
                largest_internal = next((c for c in stats['exact_4']['largest'] if not c['touches_border']),None)
                if largest_internal:
                    x0,y0,x1,y1 = largest_internal['bbox_xyxy']
                    positions=np.array([[(y0+y1)//2,(x0+x1)//2]])
            if len(positions):
                yy,xx=positions[len(positions)//2]
                left=max(0,min(rgb.shape[1]-192,int(xx)-96)); top=max(0,min(rgb.shape[0]-192,int(yy)-96))
                Image.fromarray(rgb[top:top+192,left:left+192]).resize((576,576),Image.Resampling.NEAREST).save(output/f'{rid}-{name}-detail.png')
        entry['diagnostic_hashes']={p.name:sha(p) for p in output.glob(f'{rid}-*')}
        entry['analysis_seconds']=time.perf_counter()-started
        result['samples'].append(entry)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.cache,args.output),indent=2))
