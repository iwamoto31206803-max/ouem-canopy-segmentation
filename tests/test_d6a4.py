import pytest

from test_d6a32_gate import fake_real, synthetic
import d6a32_gate as gate
import d6a3_guard
import d6a3_produce as producer
from d6a4 import D6Shards, SOURCE_FIELDS, aggregate_document, geographic_order
from r2_train import make_plan


def files_in(folder):
    return {p.relative_to(folder):gate.qa.sha(p) for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('synthetic',[(320,6)],indirect=True)
def test_pilot_and_shard_roundtrip(fake_real,monkeypatch):
    plan,write,(base,root,evidence,out),_=fake_real
    pilot=out.with_name('pilot')
    shard=out.with_name('shard')
    monkeypatch.setattr(d6a3_guard,'allocated_bytes',lambda p:p.stat().st_size)
    ref=plan['a1']['train']
    plan['sources']=[{k:s[k] for k in SOURCE_FIELDS} for s in plan['sources']]
    first=plan['sources'][:4]
    pilot_plan=dict(plan,sources=first)
    pilot_ref=write('plan.json',pilot_plan)
    raw=gate.verify_evidence(pilot_plan,pilot_ref['sha256'],base,root,evidence)
    pilot_pin=producer.publish(base/'plan.json',pilot_ref['sha256'],root,pilot,raw,real=True)['manifest_sha256']
    roster=dict(schema=gate.ROSTER,pilot_sha256=pilot_pin,sources=plan['sources'],shard_sizes=[2])
    roster_ref=write('roster.json',roster)
    shard_plan=dict(plan,schema=gate.SHARD,a1=[ref],roster=roster_ref,shard_start=4,sources=plan['sources'][4:])
    shard_ref=write('plan.json',shard_plan)
    token=write('approval.json',dict(schema=gate.APPROVAL,approved=True,
                                    plan_sha256=shard_ref['sha256'],purpose=gate.SHARD_PURPOSE))
    assert gate.approval(base/'plan.json',shard_ref['sha256'],base/'approval.json',token['sha256'])==shard_plan
    raw=gate.verify_evidence(shard_plan,shard_ref['sha256'],base,root,evidence)
    shard_pin=producer.publish(base/'plan.json',shard_ref['sha256'],root,shard,raw,real=True)['manifest_sha256']
    before=files_in(pilot),files_in(shard)
    assert producer.publish(base/'plan.json',shard_ref['sha256'],root,shard,raw,real=True,resume=True)['skipped_sources']==2
    assert (files_in(pilot),files_in(shard))==before
    parts=[dict(path=str(pilot),sha256=pilot_pin),dict(path=str(shard),sha256=shard_pin)]
    def open_dataset(parts, count=6):
        doc=aggregate_document(roster_ref,roster,parts,count)
        link=write('dataset.json',doc)
        return D6Shards(base/'dataset.json',link['sha256'])
    data=open_dataset(parts)
    assert data.complete and len(data.samples)==6
    schedule=make_plan(data,steps=3)
    for role in ('train','val'):
        assert len({s['image_id'] for s in schedule[role]})==3
        for item in schedule[role]:
            assert data.patch(item,role)[0].shape==(1,3,224,224)
            with pytest.raises(ValueError,match='D6_PATCH_IDENTITY'):
                data.patch(item,'train' if role=='val' else 'val')
    with pytest.raises(ValueError,match='INPUT_PIN_MISMATCH'):
        D6Shards(base/'dataset.json','0'*64)
    assert not open_dataset(parts[:1],4).complete
    with pytest.raises(ValueError,match='DUPLICATE_SHARD'):
        open_dataset([parts[0],parts[0]])
    with pytest.raises(ValueError,match='MISSING|Missing'):
        open_dataset([parts[0],dict(path=str(out/'missing'),sha256=shard_pin)])
    with pytest.raises(ValueError,match='ALLOWLIST'):
        open_dataset(parts[:1])
    with pytest.raises(ValueError,match='ROSTER_BINDING'):
        gate.match_roster(dict(shard_plan,shard_start=3),roster)
    second=list(roster['sources'])
    second[-1]=dict(second[-1],oam_id=second[0]['oam_id'].upper())
    with pytest.raises(ValueError,match='ROSTER_DUPLICATE'):
        gate.roster_sources(dict(roster,sources=second))
    tiff=next(shard.rglob('rgb.tif'))
    tiff.write_bytes(b'corruption')
    with pytest.raises(ValueError,match='PRODUCT_INTEGRITY'):
        open_dataset(parts)


def test_order_is_stable_and_prefers_new_groups():
    rows=[dict(image_id=i,region=[i%3,0],historical_train_reuse=i>6) for i in range(9)]
    order=geographic_order(rows)
    assert order==geographic_order(rows[::-1])
    assert len({tuple(r['region']) for r in order[:3]})==3
    assert {r['image_id'] for r in order[:7]}==set(range(7))
