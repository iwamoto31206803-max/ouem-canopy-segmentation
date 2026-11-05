from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_5a_inventory import role,eligibility,split_audit,select_candidates


def row(i,group,fold=1,split='train',lon=0,license='CC-BY 4.0'):
    return dict(image_id=i,oam_id=group,validation_fold=fold,official_split=split,license=license,
        height=2048,width=2048,crs='EPSG:3395',bounds=[0,0,1,1],meta='https://example.invalid',
        wgs84_bounds=[lon,0,lon+.001,.001],center=[lon,0],biome=i%3)


def originals(): return [row(661,'oldtrain',2,lon=-100),row(1972,'oldval',0,lon=-90)]


def test_duplicate_oam_within_role_counted():
    result=split_audit(originals()+[row(1,'new',1,lon=10),row(2,'new',1,lon=11)])
    assert result['duplicate_oam_groups']==1 and result['unique_oam_ids']==3


@pytest.mark.parametrize('split,fold',[('train',0),('test',-1)])
def test_cross_role_or_holdout_rejected(split,fold):
    rows=originals()+[row(1,'new',1),row(2,'new',fold,split)]
    with pytest.raises(ValueError): split_audit(rows)


def test_existing_roles_must_remain():
    rows=originals(); rows[0]['validation_fold']=0
    with pytest.raises(ValueError,match='Existing role'): split_audit(rows)


def test_missing_and_unknown_license_hold():
    r=row(1,'new'); r['license']=None
    assert eligibility(r)=='HOLD_MISSING_METADATA'
    r['license']='unknown'; assert eligibility(r)=='HOLD_LICENSE'
    r['license']='CC-BY 4.0'; r.pop('crs'); assert eligibility(r)=='HOLD_MISSING_METADATA'


def test_deterministic_selection_exclusions_and_holdout():
    rows=originals()+[row(i,'g'+str(i),0 if i%3==0 else 1,lon=i*2) for i in range(1,15)]
    rows.append(row(100,'held',-1,'test',lon=50))
    a,_=select_candidates(rows,4,2); b,_=select_candidates(list(reversed(rows)),4,2)
    assert [r['image_id'] for r in a]==[r['image_id'] for r in b]
    assert len(a)==6 and len({r['oam_id'] for r in a})==6
    assert not {661,1972,100}&{r['image_id'] for r in a}
    assert all(role(r)!='holdout' for r in a)


def test_opposite_role_spatial_overlap_held():
    rows=originals()+[row(1,'newtrain',1,lon=10),row(2,'newval',0,lon=10)]
    selected,reasons=select_candidates(rows,1,1)
    assert selected==[]
    assert reasons[1]=='HOLD_CROSS_ROLE_FOOTPRINT_OVERLAP'


def test_existing_identity_and_license_preserved():
    from m1_5a_inventory import verify_existing
    rows=originals()
    established=[dict(r,split=role(r)) for r in rows]
    assert verify_existing(rows,established)
    established[0]['oam_id']='different'
    with pytest.raises(ValueError,match='identity'): verify_existing(rows,established)
