"""Deterministic selection of distinct OAM source groups."""
import hashlib

def select_records(rows, train_count=4, validation_count=2, validation_fold=0):
    eligible = [r for r in rows if r.get("split") == "train"]
    eligible.sort(key=lambda r: (hashlib.sha256(str(r["oam_id"]).encode()).hexdigest(), str(r["image_id"])))
    result = {"train": [], "validation": []}
    used = set()
    for split, count in (("validation", validation_count), ("train", train_count)):
        for row in eligible:
            if len(result[split]) == count:
                break
            if not row.get("oam_id") or row.get("validation_fold") is None:
                raise ValueError("missing OAM group or fold")
            matches = int(row["validation_fold"]) == validation_fold
            if matches != (split == "validation") or row["oam_id"] in used:
                continue
            result[split].append(row)
            used.add(row["oam_id"])
        if len(result[split]) != count:
            raise ValueError("not enough distinct OAM groups")
    return result
