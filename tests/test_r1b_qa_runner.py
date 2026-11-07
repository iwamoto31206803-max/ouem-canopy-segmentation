"""Synthetic images, offline approvals and mock HTTP only; no production receipts."""
import copy
import io
import json
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import r1b_qa_runner as q
import r1b_campaign as c


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('REAL NETWORK FORBIDDEN IN OFFLINE TESTS')
    monkeypatch.setattr(socket,'create_connection',fail)
    monkeypatch.setattr(socket.socket,'connect',fail)


class Response:
    def __init__(self,body,status=200,headers=None,fail_after=None):
        self.status=status;self.body=io.BytesIO(body);self.fail_after=fail_after;self.reads=[]
        self.headers=headers if headers is not None else [('Content-Length',str(len(body))),('Content-Type','application/geo+json')]
    def getheaders(self):return self.headers
    def getheader(self,key,default=None):return next((v for k,v in self.headers if k.lower()==key.lower()),default)
    def read(self,n):
        self.reads.append(n)
        if self.fail_after is not None:
            if self.body.tell()>=self.fail_after:raise OSError('mock interrupted')
            n=min(n,self.fail_after-self.body.tell())
        return self.body.read(n)


class Factory:
    def __init__(self,response):self.response=response;self.instances=[];self.requests=[]
    def __call__(self,host,**kwargs):
        parent=self
        class Connection:
            def request(self,*args,**kw):parent.requests.append((args,kw))
            def getresponse(self):return parent.response
            def close(self):self.closed=True
        connection=Connection();connection.host=host;connection.closed=False;self.instances.append(connection);return connection


@pytest.fixture
def data(tmp_path):
    cache=tmp_path/'cache';archives=cache/'archives';archives.mkdir(parents=True)
    (cache/'SYNTHETIC_QA_FIXTURE.txt').write_bytes(q.FIXTURE_MARKER)
    base=cache/'extracted/dataset/holdout';(base/'images').mkdir(parents=True);(base/'masks').mkdir()
    rows=[];pool=[]
    for i in range(25):
        transform=from_origin(i*100000,10000,.1,.1);bounds=[i*100000,9999.2,i*100000+.8,10000]
        row=dict(image_id=i,oam_id=f'source-{i}',validation_fold=0 if i%5==4 else 1,license='CC-BY 4.0',
            width=8,height=8,crs='EPSG:3395',bounds=bounds,biome=i%3,meta=f'https://example.org/source/{i}_meta.json')
        rows.append(row);pool.append(dict(row,official_split='train',split='val' if i%5==4 else 'train'))
        rgb=np.full((3,8,8),100+i,np.uint8)
        with rasterio.open(base/'images'/f'source-{i}_{i}.tif','w',driver='GTiff',width=8,height=8,count=3,dtype='uint8',crs='EPSG:3395',transform=transform) as dst:dst.write(rgb)
        ann=np.zeros((8,8,3),np.uint8);ann[:,4:,0]=1;Image.fromarray(ann).save(base/'masks'/f'source-{i}_{i}.png')
    (archives/'train_meta.json').write_text(''.join(json.dumps(r)+'\n' for r in rows));(archives/'test_meta.json').write_text('')
    coco=archives/'dataset/holdout/train.json';coco.parent.mkdir(parents=True)
    coco.write_text(json.dumps(dict(annotations=[dict(image_id=i,id=100+i,category_id=1,iscrowd=0) for i in range(25)])))
    plan=dict(schema='ouem-r1b1-plan-v1',source_generation=c.GENERATION,plan=dict(pool=pool,initial_batch_ids=list(range(25)),completeness=dict(status='PASS')),
        inputs=dict(metadata_sha256={n:c.sha(archives/n) for n in ('train_meta.json','test_meta.json')}))
    plan['plan_id']='ouem-r1b1-'+c.hash_bytes(c.encoded(plan));p=tmp_path/'plan.json';p.write_bytes(c.encoded(plan))
    setup=dict(root=tmp_path/'campaign',registry=tmp_path/'registry',plan_path=p,plan_sha=c.sha(p),cache=cache,campaign_id='synthetic-only')
    c.Campaign.create(**setup)
    return setup


def opened(s):return c.Campaign.open(s['root'],s['registry'],s['plan_path'],s['plan_sha'])


def approval(s,campaign,changes=None):
    contract=campaign.contract
    a=dict(schema='ouem-r1b-qa-approval-v1',approved=True,approval_id='offline-fixture',reviewer_ref='synthetic test, not human production approval',mode='offline_test',
        campaign_id=contract['campaign_id'],contract_sha256=campaign._contract_sha,plan_id=contract['plan_id'],plan_sha256=contract['plan_sha256'],targets=contract['targets'],
        limits=contract['limits'],cache_root=contract['cache_root'],output_root=str(campaign.root.resolve()),environment=q.environment(),
        scopes=['qa_local','qa_preview','qa_report','stac_network','geotiff_adapter'],
        artifact_limits=dict(preview=1024**2,report=131072,stac=65536,geotiff=1024**2),max_input_pixels=64,max_buffer_bytes=2*1024**2,
        local_stac={},coco_sha256=c.sha(s['cache']/'archives/dataset/holdout/train.json'))
    a['test_fixture_sha256']=c.hash_bytes(q.FIXTURE_MARKER)
    if changes:a.update(changes)
    path=s['root'].parent/'offline-approval.json';path.write_bytes(c.encoded(a))
    return path,c.sha(path)


def stac_body(image_id=0,**updates):
    item=dict(type='Feature',stac_version='1.1.0',id=f'source-{image_id}',bbox=[-179,-80,179,80],geometry=dict(type='Polygon',coordinates=[[[-179,-80],[179,-80],[179,80],[-179,80],[-179,-80]]]),
        assets=dict(metadata=dict(href=f'https://example.org/source/{image_id}_meta.json'),visual=dict(href=f'https://example.org/source/{image_id}.tif')),
        properties=dict(license='CC-BY-4.0',providers=[dict(name='Synthetic provider',roles=['licensor','producer'])],**{'oam:producer_name':'Synthetic provider'}))
    item.update(updates);return c.encoded(item)


def runner(s,campaign,factory=None,changes=None):
    path,digest=approval(s,campaign,changes)
    return q.Runner(campaign,s['plan_path'],path,digest,test_mode=True,connection_factory=factory or Factory(Response(stac_body())))


def test_unapproved_runner_never_opens_images_or_http(data,monkeypatch):
    s=data
    with opened(s) as campaign:
        def fail(*a,**k):raise AssertionError('UNAPPROVED SOURCE ACCESS')
        monkeypatch.setattr(q,'local_index',fail)
        with pytest.raises(c.Stop,match='APPROVAL_REQUIRED'):q.Runner(campaign,s['plan_path'])
    with pytest.raises(SystemExit):q.main(['run'])


def test_offline_mode_requires_synthetic_marker(data):
    s=data
    with opened(s) as campaign:
        path,digest=approval(s,campaign);(s['cache']/'SYNTHETIC_QA_FIXTURE.txt').unlink()
        with pytest.raises(c.Stop,match='FIXTURE_MARKER'):q.Runner(campaign,s['plan_path'],path,digest,test_mode=True,connection_factory=Factory(Response(stac_body())))


def test_offline_mode_rejects_live_connection_class(data):
    s=data
    with opened(s) as campaign:
        path,digest=approval(s,campaign)
        with pytest.raises(c.Stop,match='LIVE_TRANSPORT'):q.Runner(campaign,s['plan_path'],path,digest,test_mode=True,connection_factory=q.DirectHTTPS)


@pytest.mark.parametrize('field',['campaign_id','plan_id','plan_sha256','targets','limits','output_root','environment','approved'])
def test_approval_binding_and_tampering(data,field):
    s=data
    with opened(s) as campaign:
        path,digest=approval(s,campaign);a=json.loads(path.read_bytes());a[field]=False if field=='approved' else 'wrong'
        path.write_bytes(c.encoded(a))
        with pytest.raises(c.Stop):q.Runner(campaign,s['plan_path'],path,c.sha(path),test_mode=True,connection_factory=Factory(Response(stac_body())))
        with pytest.raises(c.Stop):q.Approval(campaign,path,digest,test_mode=True)


def test_mode_scope_mutation_and_target_checks(data):
    s=data
    with opened(s) as campaign:
        r=runner(s,campaign)
        for image,split in ((25,'train'),(0,'val')):
            with pytest.raises(c.Stop):r.run_one(image,split)
        r.approval.record['scopes'].remove('stac_network')
        with pytest.raises(c.Stop,match='IN_MEMORY'):r.stac.fetch(0,'train')
        path,digest=approval(s,campaign,dict(scopes=['qa_local','qa_report','qa_preview']))
        r=q.Runner(campaign,s['plan_path'],path,digest,test_mode=True,connection_factory=Factory(Response(stac_body())))
        with pytest.raises(c.Stop,match='SCOPE'):r.run_one(0,'train',fetch_missing=True)
        assert not campaign.state['operations']
        with pytest.raises(c.Stop,match='MODE'):q.Approval(campaign,path,digest)


def test_mock_stac_single_get_limits_and_receipt(data):
    s=data;factory=Factory(Response(stac_body()))
    with opened(s) as campaign:
        r=runner(s,campaign,factory)
        result=r.stac.fetch(0,'train')
        assert result['status']=='HOLD' and result['automated_declaration_check']=='PASS'
        assert len(factory.instances)==len(factory.requests)==1 and factory.instances[0].closed
        assert factory.requests[0][0]==('GET','/stac/collections/openaerialmap/items/source-0')
        assert factory.requests[0][1]['headers']['Accept-Encoding']=='identity'
        assert all(0<n<=65536 for n in factory.response.reads)
        assert campaign.usage()['network_charged_bytes']==len(stac_body())
        assert campaign.state['operations']['stac-0']['file']['sha256']==c.hash_bytes(stac_body())
        with pytest.raises(c.Stop,match='duplicate'):r.stac.fetch(0,'train')


@pytest.mark.parametrize('url',['http://api.imagery.hotosm.org/stac/collections/openaerialmap/items/source-0',q.STAC_BASE+'source-1',q.STAC_BASE+'source-0?x=1','https://example.org/a.tif','https://example.org/a.parquet'])
def test_url_allowlist_rejected_before_reservation(data,url):
    s=data;factory=Factory(Response(stac_body()))
    with opened(s) as campaign:
        r=runner(s,campaign,factory)
        with pytest.raises(c.Stop,match='URL'):r.stac.fetch(0,'train',url)
        assert not factory.instances and not campaign.state['operations']


@pytest.mark.parametrize('problem',['redirect','retry_status','gzip','chunked','oversize','duplicate_length','duplicate_encoding','duplicate_type','no_length','truncated','connection_failure','partial_failure'])
def test_http_failures_hold_full_reservation_no_retry(data,problem):
    s=data;body=stac_body();response=Response(body)
    if problem=='redirect':response.status=302;response.headers.append(('Location',q.STAC_BASE+'source-1'))
    elif problem=='retry_status':response.status=503
    elif problem=='gzip':response.headers.append(('Content-Encoding','gzip'))
    elif problem=='chunked':response.headers.append(('Transfer-Encoding','chunked'))
    elif problem=='oversize':response.headers[0]=('Content-Length','65536')
    elif problem=='duplicate_length':response.headers.append(('Content-Length','4'))
    elif problem=='duplicate_encoding':response.headers.extend([('Content-Encoding','identity'),('Content-Encoding','gzip')])
    elif problem=='duplicate_type':response.headers.append(('Content-Type','image/tiff'))
    elif problem=='no_length':response.headers=response.headers[1:]
    elif problem=='truncated':response.headers[0]=('Content-Length',str(len(body)+1))
    elif problem=='partial_failure':response.fail_after=3
    factory=Factory(response)
    if problem=='connection_failure':
        def connect_fail(*a,**k):raise OSError('mock connect failure')
        factory=connect_fail
    with pytest.raises((c.Stop,OSError)):
        with opened(s) as campaign:
            r=runner(s,campaign,factory);r.stac.fetch(0,'train')
    report=c.Campaign.inspect(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
    assert report['status']=='STOP' and report['usage']['requests']==1
    assert report['usage']['network_charged_bytes']==65536
    if problem!='connection_failure':assert len(factory.instances)==len(factory.requests)==1
    if problem=='partial_failure':assert report['usage']['known_received_bytes']==3


def test_real_http_response_has_unbuffered_body_no_auto_request():
    packet=b'HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc'
    class Socket:
        calls=[]
        def makefile(self,*args,**kwargs):self.calls.append(kwargs);return io.BytesIO(packet)
    sock=Socket();response=q.NoPrefetchResponse(sock);assert sock.calls[-1]=={'buffering':0}
    response.begin();assert response.read(2)==b'ab' and response.read(1)==b'c' and response.read(1)==b''
    assert q.DirectHTTPS.response_class is q.NoPrefetchResponse


def test_local_qa_preview_reports_and_r1_hold(data):
    s=data
    with opened(s) as campaign:
        r=runner(s,campaign)
        report=r.run_one(0,'train')
        assert report['technical_status']=='PASS' and report['r1_acceptance']=='HOLD'
        assert report['checks']['labels']['canopy_fraction']==.5
        assert report['visual_qa']==report['external_independence']=='NOT_VERIFIED'
        assert {'source','rights','category_review','validity_review','independence_review','external_independence_review','r1_review'}<=set(report['missing_evidence'])
        assert report['source_generation']==c.GENERATION
        for op in campaign.state['operations'].values():
            assert op['status']=='SUCCESS'
            path=campaign.root/op['output_path'];assert c.sha(path)==op['file']['sha256']
        assert campaign.usage()['requests']==0
        with pytest.raises(c.Stop,match='duplicate'):r.run_one(0,'train')


@pytest.mark.parametrize('value',[0,1])
def test_uniform_labels_are_preserved(data,value):
    s=data;array=np.zeros((8,8,3),np.uint8);array[:,:,0]=value
    Image.fromarray(array).save(s['cache']/'extracted/dataset/holdout/masks/source-0_0.png')
    with opened(s) as campaign:
        r=runner(s,campaign);report=r.run_one(0,'train')
        assert report['checks']['labels']['canopy_fraction']==value
        assert report['r1_acceptance']=='HOLD'


def test_memory_buffer_hold_has_charged_report_and_pending_reservation(data):
    s=data
    with pytest.raises(q.BufferLimit):
        with opened(s) as campaign:
            r=runner(s,campaign,changes=dict(artifact_limits=dict(preview=16,report=131072,stac=65536,geotiff=1024**2)))
            r.run_one(0,'train')
    report=c.Campaign.inspect(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
    assert report['status']=='STOP' and report['usage']['unresolved_storage_reservation_bytes']>0
    diagnostic=json.loads((s['root']/'artifacts/reports/qa-preview-0-hold.json').read_bytes())
    assert diagnostic['status']=='HOLD' and diagnostic['observed_bytes']>diagnostic['limit']


def test_hold_diagnostic_cannot_exceed_approved_report_cap(data):
    s=data
    with pytest.raises(c.Stop,match='DIAGNOSTIC_DOES_NOT_FIT'):
        with opened(s) as campaign:
            r=runner(s,campaign,changes=dict(artifact_limits=dict(preview=16,report=16,stac=65536,geotiff=1024**2)))
            r.run_one(0,'train')
    view=c.Campaign.inspect(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
    assert view['status']=='STOP' and not (s['root']/'artifacts/reports/qa-preview-0-hold.json').exists()


def test_memoryfile_geotiff_adapter_no_disk_encoder_bypass(data):
    s=data
    with opened(s) as campaign:
        r=runner(s,campaign);a=np.ones((1,4,4),np.float32)
        receipt=r.saved.geotiff('future-demo',0,'train',a,'EPSG:3395',from_origin(0,10,.5,.5))
        with rasterio.open(campaign.root/receipt['path']) as src:assert np.array_equal(src.read(),a)
        assert receipt['native_memory_peak']=='NOT_MEASURED'


def test_wrong_labels_and_black_border_are_not_accepted(data):
    s=data
    path=s['cache']/'extracted/dataset/holdout/images/source-0_0.tif'
    with rasterio.open(path,'r+') as dst:
        arr=dst.read();arr[:,:,0:2]=0;dst.write(arr)
    with opened(s) as campaign:
        report=runner(s,campaign).run_one(0,'train')
        assert report['technical_status']=='HOLD' and report['r1_acceptance']=='HOLD'
        assert report['checks']['validity']['black_components']['border_connected_pixels']>0


@pytest.mark.parametrize('problem',['id','source','license','geometry','json'])
def test_stac_content_and_rights_fail_without_r1_pass(data,problem):
    s=data;item=json.loads(stac_body())
    if problem=='id':item['id']='source-4'
    elif problem=='source':item['assets']['metadata']['href']='https://example.org/wrong_meta.json'
    elif problem=='license':item['properties']['license']='CC-BY-NC-4.0'
    elif problem=='geometry':item['geometry']['coordinates']=[]
    body=b'not JSON' if problem=='json' else c.encoded(item)
    with opened(s) as campaign:
        r=runner(s,campaign,Factory(Response(body)))
        report=r.run_one(0,'train',fetch_missing=True)
        assert report['evidence']['rights']['status']=='HOLD' and report['r1_acceptance']=='HOLD'
        assert campaign.usage()['requests']==1
        assert campaign.usage()['network_charged_bytes']==len(body)


@pytest.mark.parametrize('problem',['category','dimensions','grid','dtype','png_bit_depth'])
def test_source_label_geometry_failures_are_recorded(data,problem):
    s=data
    if problem=='category':
        path=s['cache']/'archives/dataset/holdout/train.json';record=json.loads(path.read_text());record['annotations'][0]['category_id']=3;path.write_text(json.dumps(record))
    elif problem=='dimensions':Image.fromarray(np.zeros((4,4,3),np.uint8)).save(s['cache']/'extracted/dataset/holdout/masks/source-0_0.png')
    elif problem=='png_bit_depth':
        path=s['cache']/'extracted/dataset/holdout/masks/source-0_0.png';raw=bytearray(path.read_bytes());raw[24]=16;path.write_bytes(raw)
    elif problem=='dtype':
        path=s['cache']/'extracted/dataset/holdout/images/source-0_0.tif'
        with rasterio.open(path) as src:profile=src.profile;array=src.read().astype(np.uint16)
        profile['dtype']='uint16'
        with rasterio.open(path,'w',**profile) as dst:dst.write(array)
    else:
        path=s['cache']/'extracted/dataset/holdout/images/source-0_0.tif'
        with rasterio.open(path,'r+') as dst:dst.transform=from_origin(10,10000,.1,.1)
    with opened(s) as campaign:
        report=runner(s,campaign).run_one(0,'train')
        assert report['technical_status']=='FAIL' and report['r1_acceptance']=='HOLD'
        assert (campaign.root/'artifacts/reports/qa-report-0.json').is_file()


def test_save_budget_rejected_before_encoder_and_interrupted_write_stops(data,monkeypatch):
    s=data
    with opened(s) as campaign:
        r=runner(s,campaign);called=[]
        # Available disk cannot hold the frozen reservation; encoder must not run.
        class Space:free=0
        original=c.shutil.disk_usage;monkeypatch.setattr(c.shutil,'disk_usage',lambda p:Space())
        with pytest.raises(c.Stop):r.saved.encode('reject',0,'train','preview','artifacts/reject.png',lambda b:called.append(True))
        assert not called and 'reject' not in campaign.state['operations']
        monkeypatch.setattr(c.shutil,'disk_usage',original)
    original_write=c.durable
    def partial(path,data):
        if path.suffix=='.partial':path.write_bytes(data[:2]);raise OSError('mock partial write')
        return original_write(path,data)
    monkeypatch.setattr(c,'durable',partial)
    with pytest.raises(OSError):
        with opened(s) as campaign:runner(s,campaign).saved.report('partial-report',0,'train',dict(status='HOLD'))
    view=c.Campaign.inspect(s['root'],s['registry'],s['plan_path'],s['plan_sha'])
    assert view['status']=='STOP' and view['partial_files'][0]['bytes']==2


def test_coco_and_metadata_pins_and_unapproved_cli(data,monkeypatch):
    s=data
    with opened(s) as campaign:
        path,digest=approval(s,campaign)
        coco=s['cache']/'archives/dataset/holdout/train.json';original=coco.read_bytes();coco.write_bytes(b'changed')
        with pytest.raises(c.Stop,match='COCO_SHA'):q.Runner(campaign,s['plan_path'],path,digest,test_mode=True,connection_factory=Factory(Response(stac_body())))
        coco.write_bytes(original)
    # Even a correctly bound offline receipt is not usable through production CLI.
    with pytest.raises(c.Stop,match='MODE'):
        q.main(['run','--execute','--root',str(s['root']),'--registry',str(s['registry']),'--plan',str(s['plan_path']),
                '--plan-sha256',s['plan_sha'],'--approval',str(path),'--approval-sha256',digest])


def test_only_pinned_or_campaign_committed_stac_inputs_are_read(data):
    s=data;folder=s['root'].parent/'input-evidence';folder.mkdir();path=folder/'source-0-stac.json';path.write_bytes(stac_body())
    with opened(s) as campaign:
        r=runner(s,campaign)
        with pytest.raises(c.Stop,match='PATH_NOT_APPROVED'):r.check_stac(0,'train',path,c.sha(path))
        changes=dict(local_stac={'0':dict(path=str(path.resolve()),sha256=c.sha(path))})
        r=runner(s,campaign,changes=changes);result=r.run_one(0,'train')
        assert result['evidence']['rights']['status']=='HOLD' and result['evidence']['rights']['automated_declaration_check']=='PASS'
        assert result['checks']['stac_extent']['status']=='PASS'
        assert result['evidence']['rights']['source_hashes']['image_sha256']==result['evidence']['source']['image_sha256']
        assert result['r1_acceptance']=='HOLD' and campaign.usage()['requests']==0


def test_cli_summary_via_explicit_offline_test_injection(data,monkeypatch):
    s=data
    with opened(s) as campaign:path,digest=approval(s,campaign)
    original_approval=q.Approval;original_runner=q.Runner
    # Only the test harness injects test mode. Production CLI exposes no such flag.
    monkeypatch.setattr(q,'Approval',lambda campaign,path,digest,**kw:original_approval(campaign,path,digest,test_mode=True))
    monkeypatch.setattr(q,'Runner',lambda campaign,plan,path,digest:original_runner(campaign,plan,path,digest,test_mode=True,connection_factory=Factory(Response(stac_body()))))
    result=q.main(['run','--execute','--root',str(s['root']),'--registry',str(s['registry']),'--plan',str(s['plan_path']),
        '--plan-sha256',s['plan_sha'],'--approval',str(path),'--approval-sha256',digest,'--ids','0','4'])
    assert result==2
    summary=json.loads((s['root']/'artifacts/reports/qa-summary.json').read_bytes())
    assert summary['requested_ids']==[0,4] and len(summary['not_processed_ids'])==23
    assert summary['r1_acceptance']=='HOLD' and summary['usage_snapshot_before_summary']['requests']==0
