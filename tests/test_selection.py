import pytest
from canopy_prep.selection import select_records

def test_deterministic_group_disjoint():
    rows = [{"split":"train","image_id":i,"oam_id":f"site{i}","validation_fold":i%5} for i in range(35)]
    first = select_records(rows)
    assert first == select_records(list(reversed(rows)))
    assert len(first["train"]) == 4
    assert len(first["validation"]) == 2
    assert not ({r["oam_id"] for r in first["train"]} & {r["oam_id"] for r in first["validation"]})

def test_holdout_excluded():
    rows = [{"split":"test","image_id":i,"oam_id":str(i),"validation_fold":0} for i in range(10)]
    with pytest.raises(ValueError):
        select_records(rows)
