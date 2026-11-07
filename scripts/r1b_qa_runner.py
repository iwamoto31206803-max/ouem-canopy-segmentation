"""Approval-bound QA and managed adapters. No approval issuer or training code."""
import argparse
from importlib.metadata import version
import http.client
import io
import json
import platform
import re
from pathlib import Path
import ssl
import subprocess
import uuid
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile

from r1b_campaign import Campaign, Stop, ordinary, encoded, sha, hash_bytes, GENERATION, inputs
from m1_5d_zenodo_prepare import local_index, semantic_binary
from m1_5c_zenodo_audit import coco_annotation_summary
from m1_5b_pilot import separation, rights
from m1_5b2_border_audit import component_report
from m1_2b_overlay_qa import overlay
from m1_2d_geometry_audit import intervals
from m1_2d_prepare_pairs import output_grid
from r1b_candidate_plan import identity

REPO=Path(__file__).resolve().parents[1]
STAC_BASE='https://api.imagery.hotosm.org/stac/collections/openaerialmap/items/'
SCOPES={'qa_local','qa_preview','qa_report','stac_network','geotiff_adapter'}
FIXTURE_MARKER=b'OUEM synthetic QA fixture only; not a production dataset\n'


def environment():
    return dict(python=platform.python_version(),platform=platform.platform(),
        packages={k:version(k) for k in ('numpy','Pillow','rasterio','pyproj')},
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
        runner_sha256=sha(Path(__file__)))


def strict_json(data):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise Stop('DUPLICATE_JSON_KEY')
            result[k]=v
        return result
    return json.loads(data,object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(Stop('NONFINITE_JSON')))


class Approval:
    def __init__(self,campaign,path,expected_sha,*,test_mode=False):
        if path is None or not expected_sha:raise Stop('EXPLICIT_APPROVAL_REQUIRED')
        ordinary(Path(path));data=Path(path).read_bytes()
        if hash_bytes(data)!=expected_sha:raise Stop('APPROVAL_SHA_MISMATCH')
        a=strict_json(data);c=campaign.contract
        expected=dict(campaign_id=c['campaign_id'],contract_sha256=campaign._contract_sha,
            plan_id=c['plan_id'],plan_sha256=c['plan_sha256'],targets=c['targets'],
            limits=c['limits'],cache_root=c['cache_root'],output_root=str(campaign.root.resolve()),environment=environment())
        if a.get('schema')!='ouem-r1b-qa-approval-v2' or a.get('approved') is not True or not a.get('approval_id') or not a.get('reviewer_ref'):
            raise Stop('INVALID_APPROVAL_RECORD')
        if any(encoded(a.get(k))!=encoded(v) for k,v in expected.items()):raise Stop('APPROVAL_BINDING_MISMATCH')
        if a.get('mode')!=('offline_test' if test_mode else 'production_qa'):raise Stop('APPROVAL_MODE_MISMATCH')
        if test_mode:
            marker=Path(c['cache_root'])/'SYNTHETIC_QA_FIXTURE.txt';ordinary(marker)
            if not marker.is_file() or marker.read_bytes()!=FIXTURE_MARKER or a.get('test_fixture_sha256')!=hash_bytes(FIXTURE_MARKER):raise Stop('SYNTHETIC_FIXTURE_MARKER_REQUIRED')
        if not test_mode and a['environment']['git_dirty']:raise Stop('CLEAN_PINNED_COMMIT_REQUIRED')
        scopes=a.get('scopes')
        if not isinstance(scopes,list) or len(scopes)!=len(set(scopes)) or not set(scopes)<=SCOPES:raise Stop('INVALID_SCOPE')
        allowed=a.get('allowed_image_ids')
        if not isinstance(allowed,list) or not allowed or any(type(i) is not int or str(i) not in c['targets'] for i in allowed) or len(set(allowed))!=len(allowed):raise Stop('INVALID_ALLOWED_IMAGE_IDS')
        limits=a.get('artifact_limits',{})
        if set(limits)!={'preview','report','stac','geotiff'} or any(type(v) is not int or not 0<v<=32*1024**2 for v in limits.values()):raise Stop('INVALID_ARTIFACT_LIMITS')
        if limits['stac']>c['limits']['response_bytes']:raise Stop('STAC_BUFFER_LIMIT')
        for key,ceiling in [('max_input_pixels',2048**2),('max_buffer_bytes',32*1024**2)]:
            if type(a.get(key)) is not int or not 0<a[key]<=ceiling:raise Stop('INVALID_MEMORY_GUARD')
        if not set(a.get('local_stac',{}))<=set(c['targets']):raise Stop('UNPLANNED_STAC_INPUT')
        for proof in a.get('local_stac',{}).values():
            if not isinstance(proof,dict) or not isinstance(proof.get('path'),str) or not Path(proof['path']).is_absolute() or not isinstance(proof.get('sha256'),str) or len(proof['sha256'])!=64 or any(x not in '0123456789abcdef' for x in proof['sha256']):raise Stop('INVALID_STAC_INPUT_PIN')
        self.record=a;self.sha256=expected_sha;self.path=Path(path);self.record_sha=hash_bytes(encoded(a))

    def require(self,scope):
        if sha(self.path)!=self.sha256:raise Stop('APPROVAL_CHANGED')
        if hash_bytes(encoded(self.record))!=self.record_sha:raise Stop('APPROVAL_CHANGED_IN_MEMORY')
        if scope not in self.record['scopes']:raise Stop('SCOPE_NOT_APPROVED:'+scope)

    def require_image(self,image_id):
        self.require('qa_local')
        if type(image_id) is not int or image_id not in self.record['allowed_image_ids']:raise Stop('IMAGE_NOT_IN_EXECUTION_APPROVAL')

    def select_ids(self,order,requested=None):
        self.require('qa_local')
        ids=[i for i in order if i in self.record['allowed_image_ids']] if requested is None else list(requested)
        if not ids or len(set(ids))!=len(ids):raise Stop('EMPTY_OR_DUPLICATE_IDS')
        for image_id in ids:self.require_image(image_id)
        return ids


class BufferLimit(Stop):
    def __init__(self,observed,limit):
        super().__init__(f'ENCODED_BUFFER_HOLD: observed={observed} limit={limit}');self.observed=observed;self.limit=limit


class BoundedBuffer(io.BytesIO):
    def __init__(self,limit):super().__init__();self.limit=limit
    def write(self,data):
        size=max(self.getbuffer().nbytes,self.tell()+len(data))
        if size>self.limit:raise BufferLimit(size,self.limit)
        return super().write(data)


class Saved:
    def __init__(self,campaign,approval):self.c=campaign;self.a=approval

    def _hold(self,op,image_id,split,exc):
        self.a.require_image(image_id)
        # Optional diagnostic is itself charged. Original reservation stays pending.
        data=encoded(dict(status='HOLD',operation_id=op,reason=type(exc).__name__,
            observed_bytes=getattr(exc,'observed',None),limit=getattr(exc,'limit',None),memory_peak='NOT_MEASURED'))
        self.a.require('qa_report')
        cap=min(16384,self.a.record['artifact_limits']['report'],self.a.record['max_buffer_bytes'])
        if len(data)>cap:raise Stop('HOLD_DIAGNOSTIC_DOES_NOT_FIT_APPROVED_CAP')
        key=op+'-hold';self.c.reserve(key,image_id,split,'report',f'artifacts/reports/{key}.json',cap)
        self.c.write_bytes(key,data);self.c.commit(key)

    def encode(self,op,image_id,split,kind,path,encoder):
        self.a.require_image(image_id)
        self.a.require({'preview':'qa_preview','report':'qa_report','geotiff':'geotiff_adapter'}[kind])
        cap=min(self.a.record['artifact_limits'][kind],self.a.record['max_buffer_bytes'])
        self.c.reserve(op,image_id,split,kind,path,cap)
        try:
            with BoundedBuffer(cap) as buf:
                metrics=encoder(buf) or {};data=buf.getvalue()
            self.c.write_bytes(op,data);self.c.commit(op)
            return dict(path=path,bytes=len(data),sha256=hash_bytes(data),**metrics)
        except (BufferLimit,MemoryError) as exc:
            self._hold(op,image_id,split,exc);raise

    def report(self,op,image_id,split,value):
        self.a.require_image(image_id)
        def encode(buf):
            for text in json.JSONEncoder(sort_keys=True,ensure_ascii=True,allow_nan=False).iterencode(value):buf.write(text.encode('utf-8'))
        return self.encode(op,image_id,split,'report',f'artifacts/reports/{op}.json',encode)

    def preview(self,op,image_id,split,rgb,mask):
        self.a.require_image(image_id)
        def encode(buf):
            canvas=Image.new('RGB',(rgb.shape[1]*3,rgb.shape[0]))
            for i,array in enumerate((rgb,np.repeat((mask*255)[:,:,None],3,axis=2),overlay(rgb,mask))):canvas.paste(Image.fromarray(array),(i*rgb.shape[1],0))
            canvas.thumbnail((1800,600));canvas.save(buf,format='JPEG',quality=92)
        return self.encode(op,image_id,split,'preview',f'artifacts/previews/{op}.jpg',encode)

    def geotiff(self,op,image_id,split,array,crs,transform):
        self.a.require_image(image_id)
        def encode(buf):
            estimate=2*array.nbytes+65536
            if estimate>self.a.record['max_buffer_bytes']:raise BufferLimit(estimate,self.a.record['max_buffer_bytes'])
            if array.ndim!=3 or array.dtype not in (np.uint8,np.float32) or not np.isfinite(array).all():raise Stop('INVALID_RASTER_ARRAY')
            with MemoryFile() as memory:
                with memory.open(driver='GTiff',height=array.shape[1],width=array.shape[2],count=array.shape[0],dtype=array.dtype,crs=crs,transform=transform,compress='deflate') as dst:dst.write(array)
                memory.seek(0,2);size=memory.tell()
                if size>buf.limit:raise BufferLimit(size,buf.limit)
                memory.seek(0);buf.write(memory.read())
            return dict(array_bytes=array.nbytes,native_memory_peak='NOT_MEASURED')
        return self.encode(op,image_id,split,'geotiff',f'artifacts/geotiff/{op}.tif',encode)


def stac_structure(data,row):
    item=strict_json(data)
    if not isinstance(item,dict) or item.get('type')!='Feature' or not str(item.get('stac_version','')).startswith('1.') or item.get('id')!=row['oam_id']:
        raise Stop('STAC_STRUCTURE_OR_ID')
    b=item.get('bbox')
    if not isinstance(b,list) or len(b)!=4 or not all(type(v) in (int,float) and np.isfinite(v) for v in b) or not (-180<=b[0]<b[2]<=180 and -90<=b[1]<b[3]<=90):raise Stop('STAC_BBOX')
    geometry=item.get('geometry')
    if not isinstance(geometry,dict) or geometry.get('type') not in ('Polygon','MultiPolygon'):raise Stop('STAC_GEOMETRY')
    coordinates=geometry.get('coordinates')
    polygons=[coordinates] if geometry['type']=='Polygon' else coordinates
    if not isinstance(polygons,list) or not polygons:raise Stop('STAC_GEOMETRY')
    for polygon in polygons:
        if not isinstance(polygon,list) or not polygon:raise Stop('STAC_GEOMETRY')
        for ring in polygon:
            if not isinstance(ring,list) or len(ring)<4 or ring[0]!=ring[-1]:raise Stop('STAC_GEOMETRY')
            for point in ring:
                if not isinstance(point,list) or len(point) not in (2,3) or not all(type(v) in (int,float) and np.isfinite(v) for v in point) or not -180<=point[0]<=180 or not -90<=point[1]<=90:raise Stop('STAC_GEOMETRY')
    if not isinstance(item.get('assets'),dict) or not isinstance(item.get('properties'),dict):raise Stop('STAC_STRUCTURE')
    for name in ('metadata','visual'):
        href=item.get('assets',{}).get(name,{}).get('href','')
        if urlparse(href).scheme not in ('http','https'):raise Stop('STAC_SOURCE_ASSET')
    if urlparse(item['assets']['metadata']['href']).path!=urlparse(row['meta']).path:raise Stop('STAC_ORIGINAL_SOURCE_MISMATCH')
    return item


class NoPrefetchResponse(http.client.HTTPResponse):
    """Avoid HTTP makefile read-ahead; TLS/socket wire buffering is not byte accounting."""
    def __init__(self,sock,*args,**kwargs):
        super().__init__(sock,*args,**kwargs)
        self.fp.close()  # base constructor has not read headers/body yet
        self.fp=sock.makefile('rb',buffering=0)


class DirectHTTPS(http.client.HTTPSConnection):
    response_class=NoPrefetchResponse


class Stac:
    def __init__(self,runner):self.r=runner

    def fetch(self,image_id,split,url=None):
        self.r.approval.require_image(image_id)
        r=self.r;row=r.target(image_id,split);r.approval.require('stac_network')
        expected=STAC_BASE+row['oam_id']
        if url is not None and url!=expected:raise Stop('STAC_URL_NOT_ALLOWED')
        op=f'stac-{image_id}';cap=r.c.contract['limits']['response_bytes'];output_cap=r.approval.record['artifact_limits']['stac']
        path=f"artifacts/stac/{row['oam_id']}-stac.json"
        r.c.reserve(op,image_id,split,'stac',path,output_cap,network_bytes=cap)
        holder={};received=0
        def reader(n):
            nonlocal received
            if not holder:
                # stdlib: one explicit request, no retries/redirect/proxy machinery.
                factory=r.connection_factory or DirectHTTPS
                conn=factory('api.imagery.hotosm.org',timeout=20,context=ssl.create_default_context())
                holder['connection']=conn
                conn.request('GET',urlparse(expected).path,headers={'Accept-Encoding':'identity','Connection':'close','User-Agent':'OUEM-R1b-QA/1'})
                response=conn.getresponse();holder['response']=response
                headers=response.getheaders()
                lengths=[v for k,v in headers if k.lower()=='content-length']
                encodings=[v for k,v in headers if k.lower()=='content-encoding']
                content_types=[v for k,v in headers if k.lower()=='content-type']
                if response.status!=200 or len(lengths)!=1 or not lengths[0].isdigit():raise Stop('HTTP_STATUS_OR_LENGTH')
                if response.getheader('Transfer-Encoding') is not None or len(encodings)>1 or any(v.strip().lower()!='identity' for v in encodings):raise Stop('HTTP_UNEXPECTED_ENCODING')
                if len(content_types)!=1 or content_types[0].split(';')[0].strip().lower() not in ('application/json','application/geo+json'):raise Stop('HTTP_CONTENT_TYPE')
                length=int(lengths[0])
                if not 0<length<cap or length>output_cap:raise Stop('HTTP_BODY_LIMIT_OR_EOF_NOT_VERIFIABLE')
                holder['length']=length
            chunk=holder['response'].read(n)
            if not isinstance(chunk,bytes):raise Stop('HTTP_READER_BOUND_VIOLATION')
            if len(chunk)>n:return chunk  # Campaign logs actual overrun and enters fatal STOP
            received+=len(chunk)
            if received>holder['length'] or (not chunk and received!=holder['length']):raise Stop('HTTP_TRUNCATED_OR_LENGTH_MISMATCH')
            return chunk
        try:data=r.c.receive(op,reader)
        finally:
            if 'connection' in holder:holder['connection'].close()
        r.c.write_bytes(op,data);r.c.commit(op)
        return r.check_stac(image_id,split,r.c.root/path,sha(r.c.root/path))


class Runner:
    def __init__(self,campaign,plan_path,approval_path=None,approval_sha=None,*,test_mode=False,connection_factory=None):
        self.c=campaign;campaign._ready()
        self.approval=Approval(campaign,approval_path,approval_sha,test_mode=test_mode)
        self.approval.require('qa_local')
        if connection_factory is not None and not test_mode:raise Stop('MOCK_TRANSPORT_ONLY_IN_OFFLINE_TEST')
        if test_mode and connection_factory is None:raise Stop('OFFLINE_TEST_REQUIRES_MOCK_TRANSPORT')
        if test_mode and isinstance(connection_factory,type) and issubclass(connection_factory,http.client.HTTPConnection):raise Stop('LIVE_TRANSPORT_FORBIDDEN_IN_OFFLINE_TEST')
        self.connection_factory=connection_factory
        if sha(Path(plan_path))!=campaign.contract['plan_sha256']:raise Stop('PLAN_SHA_MISMATCH')
        cache=Path(campaign.contract['cache_root'])
        checked_plan,targets,_=inputs(plan_path,campaign.contract['plan_sha256'],cache,campaign.contract['plan_id'])
        if targets!=campaign.contract['targets']:raise Stop('FIXED_TARGETS_MISMATCH')
        self.order=list(checked_plan['plan']['initial_batch_ids'])
        self.rows=local_index(cache);self.index={r['image_id']:r for r in self.rows}
        groups={}
        for row in self.rows:groups.setdefault(row['oam_id'],[]).append(row)
        self.group_catalog=[]
        for members in groups.values():
            bounds=[r['wgs84_bounds'] for r in members]
            b=[min(x[0] for x in bounds),min(x[1] for x in bounds),max(x[2] for x in bounds),max(x[3] for x in bounds)]
            self.group_catalog.append(dict(members[0],wgs84_bounds=b,center=[(b[0]+b[2])/2,(b[1]+b[3])/2]))
        self.group_index={r['oam_id']:r for r in self.group_catalog}
        for key in self.approval.record['allowed_image_ids']:self.target(key,campaign.contract['targets'][str(key)])
        chosen=[self.index[int(k)] for k in campaign.contract['targets']]
        self.duplicate_sources={r['image_id'] for r in chosen for other in chosen if r['image_id']!=other['image_id'] and
            (r['oam_id']==other['oam_id'] or (identity(r) is not None and identity(r)==identity(other)))}
        coco=cache/'archives/dataset/holdout/train.json';ordinary(coco)
        if sha(coco)!=self.approval.record.get('coco_sha256'):raise Stop('COCO_SHA_MISMATCH')
        self.categories=coco_annotation_summary(coco,[int(i) for i in campaign.contract['targets']])
        self.saved=Saved(campaign,self.approval);self.stac=Stac(self)

    def target(self,image_id,split):
        self.c._ready()
        if type(image_id) is not int or self.c.contract['targets'].get(str(image_id))!=split:raise Stop('UNPLANNED_ID_OR_SPLIT')
        self.approval.require_image(image_id)
        row=self.index[image_id]
        if row['official_split']!='train' or ('val' if row['validation_fold']==0 else 'train')!=split:raise Stop('OFFICIAL_ROLE_MISMATCH')
        stem=f"{row['oam_id']}_{image_id}"
        expected=dict(rgb=f'extracted/dataset/holdout/images/{stem}.tif',annotation=f'extracted/dataset/holdout/masks/{stem}.png',byte_validation='NOT_VERIFIED_R1B2B_REQUIRED')
        if self.c.contract['source_files'][str(image_id)]!=expected:raise Stop('SOURCE_FILENAME_ID_MISMATCH')
        return row

    def check_stac(self,image_id,split,path,expected_sha):
        self.approval.require_image(image_id)
        row=self.target(image_id,split);self.approval.require('qa_local');ordinary(path)
        local=self.approval.record.get('local_stac',{}).get(str(image_id))
        pinned=local and path.resolve()==Path(local['path']).resolve() and expected_sha==local['sha256']
        operation=self.c.state['operations'].get(f'stac-{image_id}',{})
        fetched=operation.get('status')=='SUCCESS' and path.resolve()==(self.c.root/operation['output_path']).resolve() and expected_sha==operation['file']['sha256']
        if not (pinned or fetched):raise Stop('STAC_PATH_NOT_APPROVED_OR_CAMPAIGN_COMMITTED')
        if path.name!=row['oam_id']+'-stac.json' or path.stat().st_size>=self.c.contract['limits']['response_bytes'] or sha(path)!=expected_sha:raise Stop('STAC_LOCAL_INPUT_MISMATCH')
        stac_structure(path.read_bytes(),row)
        item,receipt=rights(dict(row,source_evidence=dict(snapshot_sha256=expected_sha,url=STAC_BASE+row['oam_id'])),path.parent)
        return dict(status='HOLD',automated_declaration_check='PASS',human_rights_review='NOT_VERIFIED',stac_sha256=expected_sha,
                    image_license=receipt['image_license'],annotation_license=receipt['annotation_license'],
                    attribution='Named licensor/producer retained in external hashed STAC; human review required',bbox=item['bbox'])

    def ensure_unstarted(self,image_id):
        self.approval.require_image(image_id)
        if any(f'{prefix}-{image_id}' in self.c.state['operations'] for prefix in ('qa-preview','qa-report')):
            raise Stop('duplicate QA operation already started')

    def run_batch(self,ids=None,*,run_id=None,fetch_missing=False):
        ids=self.approval.select_ids(self.order,ids)
        self.approval.require('qa_preview');self.approval.require('qa_report')
        if fetch_missing:self.approval.require('stac_network')
        run_id=run_id or uuid.uuid4().hex
        if not isinstance(run_id,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',run_id):raise Stop('INVALID_RUN_ID')
        for image_id in ids:
            self.target(image_id,self.c.contract['targets'][str(image_id)]);self.ensure_unstarted(image_id)
        if any(f'{prefix}-{run_id}' in self.c.state['operations'] for prefix in ('qa-run','qa-summary')):raise Stop('DUPLICATE_RUN_ID')
        binding=dict(run_id=run_id,campaign_id=self.c.contract['campaign_id'],plan_id=self.c.contract['plan_id'],
            approval_id=self.approval.record['approval_id'],approval_sha256=self.approval.sha256,
            allowed_image_ids=self.approval.record['allowed_image_ids'],requested_ids=ids)
        self.saved.report(f'qa-run-{run_id}',ids[0],self.c.contract['targets'][str(ids[0])],dict(binding,status='STARTED',r1_acceptance='HOLD'))
        results=[]
        for image_id in ids:
            result=self.run_one(image_id,self.c.contract['targets'][str(image_id)],fetch_missing=fetch_missing,run_id=run_id)
            results.append(dict(image_id=image_id,technical=result['technical_status'],r1_acceptance=result['r1_acceptance']))
        reported=[i for i in self.order if self.c.state['operations'].get(f'qa-report-{i}',{}).get('status')=='SUCCESS']
        summary=dict(binding,results=results,not_requested_this_run_ids=[i for i in self.order if i not in ids],
            completed_campaign_report_ids=reported,not_processed_ids=[i for i in self.order if i not in reported],
            technical_status='FAIL' if any(r['technical']=='FAIL' for r in results) else 'HOLD' if any(r['technical']=='HOLD' for r in results) else 'PASS',
            r1_acceptance='HOLD',usage_snapshot_before_summary=self.c.usage(),process_memory_peak='NOT_MEASURED',
            new_network_permitted='stac_network' in self.approval.record['scopes'])
        self.saved.report(f'qa-summary-{run_id}',ids[0],self.c.contract['targets'][str(ids[0])],summary)
        return summary

    def run_one(self,image_id,split,*,fetch_missing=False,run_id=None):
        # Enforce the API boundary explicitly, before target lookup or source I/O.
        self.approval.require_image(image_id)
        row=self.target(image_id,split);self.approval.require('qa_local')
        self.approval.require('qa_preview');self.approval.require('qa_report')
        if fetch_missing:self.approval.require('stac_network')
        self.ensure_unstarted(image_id)
        report=dict(image_id=image_id,split=split,run_id=run_id or uuid.uuid4().hex,technical_status='FAIL',r1_acceptance='HOLD',
                    visual_qa='NOT_VERIFIED',external_independence='NOT_VERIFIED',approval_sha256=self.approval.sha256,
                    source_generation=GENERATION,checks={},evidence={},missing_evidence=[])
        for key in ('category_review','validity_review','independence_review','external_independence_review','r1_review'):
            report['evidence'][key]=dict(status='NOT_VERIFIED',required='Source-hash-bound human/catalog review receipt')
        report['evidence']['source']=dict(status='NOT_VERIFIED',required='Original bytes and archive-member authentication')
        report['evidence']['rights']=dict(status='NOT_VERIFIED',required='Hashed STAC and human rights/attribution review')
        rgb=mask=None
        try:
            cache=Path(self.c.contract['cache_root']);files=self.c.contract['source_files'][str(image_id)]
            paths={k:cache/files[k] for k in ('rgb','annotation')}
            for path in paths.values():ordinary(path);path.resolve().relative_to(cache.resolve())
            source=dict(image_sha256=sha(paths['rgb']),annotation_sha256=sha(paths['annotation']))
            report['evidence']['source']=dict(source,status='HOLD',required='ARCHIVE_MEMBER_AUTHENTICATION',dataset='restor/tcd Zenodo',revision=GENERATION,oam_id=row['oam_id'],row_license=row.get('license'),original_source_id='metadata-object-sha256:'+str(identity(row)))
            report['checks']['source_hashes']=dict(status='PASS',**source)
            report['checks']['row_license']=dict(status='PASS' if row.get('license')=='CC-BY 4.0' else 'HOLD')
            for key in report['evidence']:
                if key!='source':report['evidence'][key]['source_hashes']=dict(source)
            with paths['annotation'].open('rb') as stream:header=stream.read(29)
            if header[:16]!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or len(header)<29 or header[24:26]!=b'\x08\x02':raise Stop('ANNOTATION_8BIT_RGB_REQUIRED')
            with Image.open(paths['rgb']) as image,Image.open(paths['annotation']) as ann:
                if image.format!='TIFF' or ann.format!='PNG' or image.mode!='RGB' or ann.mode!='RGB':raise Stop('RGB_ANNOTATION_FORMAT')
                if ann.info.get('transparency') is not None:raise Stop('ANNOTATION_TRANSPARENCY_UNVERIFIED')
                if image.size!=ann.size or image.size!=(row['width'],row['height']):raise Stop('IMAGE_LABEL_DIMENSIONS')
                if image.width*image.height>self.approval.record['max_input_pixels']:raise Stop('INPUT_PIXEL_MEMORY_HOLD')
                if any(x.getexif().get(274,1)!=1 for x in (image,ann)):raise Stop('UNAPPROVED_ORIENTATION')
                rgb=np.array(image);panoptic=np.array(ann)
            mask=semantic_binary(rgb,panoptic,self.categories[image_id]['categories'])
            report['decoded_array_bytes']=rgb.nbytes+panoptic.nbytes+mask.nbytes
            report['process_memory_peak']='NOT_MEASURED'
            report['checks']['labels']=dict(status='PASS',categories=self.categories[image_id]['categories'],
                canopy_fraction=float(mask.mean()),uniform_background=bool(np.all(mask==0)),uniform_canopy=bool(np.all(mask==1)),
                policy='upstream_nonzero_binary_with_review',per_segment_category_mapping='NOT_VERIFIED')
            with rasterio.open(paths['rgb']) as src:
                t=src.transform
                if str(src.crs)!='EPSG:3395' or src.count!=3 or src.dtypes!=('uint8',)*3 or src.shape!=(row['height'],row['width']) or t.b or t.d or t.a<=0 or t.e>=0 or not np.isfinite(list(t)).all():raise Stop('CRS_AFFINE_SHAPE_DTYPE')
                if not np.allclose(src.bounds,row['bounds'],rtol=0,atol=1e-8) or not np.array_equal(src.read().transpose(1,2,0),rgb):raise Stop('GRID_OR_CHANNEL_MISMATCH')
                valid=src.nodata is None and np.all(src.read_masks()==255) and not any(c.name=='alpha' for c in src.colorinterp)
                shape,grid,gsd=output_grid(src.crs,t,src.width,src.height)
                report['checks']['geometry']=dict(status='PASS',crs=str(src.crs),affine=list(t)[:6],bounds=list(src.bounds),
                    ground_gsd=intervals(src.crs,*(t*(src.width/2,src.height/2)),t.a,-t.e),proposed_050_shape=list(shape),proposed_ground_gsd=gsd)
            black,_,_=component_report(np.all(rgb==0,axis=2),mask)
            report['checks']['validity']=dict(status='HOLD' if not valid or black['border_connected_pixels']/mask.size>=.01 else 'PASS',
                raster_masks_all_valid=bool(valid),black_components=black,visual_all_valid='NOT_VERIFIED')
            grouped=self.group_index[row['oam_id']]
            try:report['checks']['metadata_separation']=dict(status='PASS',candidate_group_bounds_wgs84=grouped['wgs84_bounds'],
                **separation(dict(row,wgs84_bounds=grouped['wgs84_bounds'],center=grouped['center']),self.group_catalog))
            except ValueError:report['checks']['metadata_separation']=dict(status='HOLD',reason='METADATA_SOURCE_OR_SPATIAL_RISK')
            report['checks']['selected_source_groups']=dict(status='HOLD' if image_id in self.duplicate_sources else 'PASS',scope='Selected metadata source identity only')
            local=self.approval.record.get('local_stac',{}).get(str(image_id))
            report['evidence']['rights']=dict(status='HOLD',reason='NO_STAC_EVIDENCE')
            try:
                if local:report['evidence']['rights']=self.check_stac(image_id,split,Path(local['path']),local['sha256'])
                elif fetch_missing:report['evidence']['rights']=self.stac.fetch(image_id,split)
            except (Stop,ValueError,KeyError,OSError,TypeError,AttributeError):
                if any(o['status'] not in ('SUCCESS','FAILED','RECONCILED_FULL_CHARGE') for o in self.c.state['operations'].values()):raise
                report['evidence']['rights']=dict(status='HOLD',reason='STAC_RIGHTS_OR_SOURCE_UNVERIFIED')
            if 'bbox' in report['evidence']['rights']:
                geo=rasterio.warp.transform_bounds(row['crs'],4326,*row['bounds']);b=report['evidence']['rights']['bbox']
                contained=b[0]-1e-8<=geo[0]<=geo[2]<=b[2]+1e-8 and b[1]-1e-8<=geo[1]<=geo[3]<=b[3]+1e-8
                report['checks']['stac_extent']=dict(status='PASS' if contained else 'HOLD',scope='Declared bbox containment only; not valid-area proof')
            report['technical_status']='HOLD' if any(c['status']=='HOLD' for c in report['checks'].values()) else 'PASS'
        except (Stop,ValueError,KeyError,OSError) as exc:
            if any(o['status'] not in ('SUCCESS','FAILED','RECONCILED_FULL_CHARGE') for o in self.c.state['operations'].values()):raise
            report['technical_status']='HOLD' if isinstance(exc,Stop) and 'MEMORY_HOLD' in str(exc) else 'FAIL'
            report['failure']=dict(type=type(exc).__name__,reason=str(exc) if isinstance(exc,Stop) else 'SOURCE_OR_LABEL_CHECK_FAILED')
        known=report['evidence']['source']
        if 'image_sha256' in known:
            for key,value in report['evidence'].items():
                if key!='source':value['source_hashes']={k:known[k] for k in ('image_sha256','annotation_sha256')}
        report['missing_evidence']=[k for k,v in report['evidence'].items() if v['status']!='PASS']
        if report['technical_status']!='FAIL' and rgb is not None and mask is not None:
            report['preview']=self.saved.preview(f'qa-preview-{image_id}',image_id,split,rgb,mask)
        self.saved.report(f'qa-report-{image_id}',image_id,split,report)
        return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check-approval','run'])
    for name in ('root','registry','plan','approval'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--approval-sha256',required=True)
    p.add_argument('--ids',type=int,nargs='+');p.add_argument('--execute',action='store_true');p.add_argument('--fetch-missing-stac',action='store_true')
    p.add_argument('--run-id',help='Unique campaign run ID; UUID generated if omitted')
    a=p.parse_args(argv)
    if a.action=='run' and not a.execute:raise Stop('EXPLICIT_EXECUTE_FLAG_REQUIRED')
    # Read-only control inspection first. No image/HTTP before approval validation.
    view=Campaign.inspect(a.root,a.registry,a.plan,a.plan_sha256)
    if view['status']!='CLEAN_PAUSED':raise Stop('CAMPAIGN_RECONCILIATION_REQUIRED')
    class Context:pass
    context=Context();context.contract=view['contract'];context.root=a.root;context._contract_sha=sha(a.root/'contract.json')
    validated=Approval(context,a.approval,a.approval_sha256)
    if a.ids is not None:validated.select_ids([int(i) for i in context.contract['targets']],a.ids)
    if a.action=='check-approval':print(json.dumps(dict(status='APPROVAL_BINDING_CHECKED',data_execution=False)));return 0
    with Campaign.open(a.root,a.registry,a.plan,a.plan_sha256) as campaign:
        runner=Runner(campaign,a.plan,a.approval,a.approval_sha256)
        summary=runner.run_batch(a.ids,run_id=a.run_id,fetch_missing=a.fetch_missing_stac)
        print(json.dumps(dict(run_id=summary['run_id'],results=summary['results'],r1_acceptance='HOLD')))
    return 1 if summary['technical_status']=='FAIL' else 2


if __name__=='__main__':raise SystemExit(main())
