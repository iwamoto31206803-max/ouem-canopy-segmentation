"""Bounded, train-only OAM-TCD raw-pair audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import requests

REVISION = "d97d4da0ebbb6e249ae95ac5e19656babd972eb2"
CODE_REVISION = "05dab754a3fd15b83d6b7606de7310155f4e2df2"
ROOT = f"https://huggingface.co/datasets/restor/tcd/resolve/{REVISION}"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def get(self, url, headers=None):
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            if headers and response.status_code != 206:
                raise ValueError("Server did not honor byte range")
            chunks = []
            for chunk in response.iter_content(65536):
                self.used += len(chunk)
                if self.used > self.limit:
                    raise ValueError("Transfer budget exceeded")
                chunks.append(chunk)
            return b"".join(chunks)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, budget):
        self.url, self.size, self.budget, self.position = url, size, budget, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        if end - self.position > self.budget.limit - self.budget.used:
            raise ValueError("Range exceeds remaining transfer budget")
        data = self.budget.get(self.url, {"Range": f"bytes={self.position}-{end-1}"})
        if len(data) != end - self.position:
            raise ValueError("Incomplete byte range")
        self.position = end
        return data


def choose(rows, count):
    selected, groups = [], {}
    for row in rows:
        fold = row["validation_fold"]
        if fold not in range(5) or row["license"] != "CC-BY 4.0":
            continue
        split = "val" if fold == 0 else "train"
        group = row["oam_id"]
        if group in groups and groups[group] != split:
            raise ValueError("oam_id Train/Val overlap")
        groups[group] = split
        if any(r["split"] == split for r in selected):
            continue
        selected.append(dict(row, split=split))
        if len(selected) == count:
            break
    return selected


def labels(image, annotation, segments):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    if image.shape[:2] != annotation.shape[:2]:
        raise ValueError("Image/annotation dimensions differ")
    if annotation.ndim != 3 or annotation.shape[2] != 3 or annotation.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB annotation")
    rgb = annotation.astype(np.uint32)
    ids = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]
    mapping = {}
    crowds = []
    for segment in segments:
        sid, category = segment["id"], segment["category_id"]
        if sid == 0 or sid in mapping or category not in (1, 2):
            raise ValueError("Invalid segment ID/category")
        mapping[sid] = category
        if segment["iscrowd"] not in (0, 1):
            raise ValueError("Unknown crowd flag")
        if segment["iscrowd"]:
            crowds.append(sid)
    present = set(np.unique(ids).tolist())
    if present - {0} - mapping.keys():
        raise ValueError("Unmapped nonzero panoptic ID")
    mask = np.isin(ids, list(mapping)).astype(np.uint8)
    return mask, {"categories": sorted(set(mapping.values())), "zero_pixels": int((ids == 0).sum()),
                  "crowd_ids": crowds, "segments_without_pixels": sorted(mapping.keys() - present),
                  "positive_pixels": int(mask.sum())}


def save_pair(folder, row, image_bytes, annotation_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    annotation = np.array(Image.open(io.BytesIO(annotation_bytes)))
    mask, qa = labels(image, annotation, json.loads(row["segments"]))
    if image.shape[:2] != (row["height"], row["width"]):
        raise ValueError("Declared dimensions differ")
    folder.mkdir()
    files = {"rgb.bin": image_bytes, "annotation.bin": annotation_bytes,
             "segments.json": row["segments"].encode(), "coco_annotations.json": row["coco_annotations"].encode()}
    hashes = {}
    for name, data in files.items():
        (folder / name).write_bytes(data)
        hashes[name] = digest((folder / name).read_bytes())
    Image.fromarray(mask).save(folder / "candidate-mask.png")
    hashes["candidate-mask.png"] = digest((folder / "candidate-mask.png").read_bytes())
    return hashes, qa


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=32)
    parser.add_argument("--transfer-mib", type=int, default=128)
    parser.add_argument("--storage-mib", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.scan_limit <= 64 or not 1 <= args.transfer_mib <= 256 or not 1 <= args.storage_mib <= 64:
        parser.error("Limits exceed bounded audit range")
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents or output.exists():
        parser.error("Use a fresh destination outside the repository")
    output.mkdir(parents=True)
    budget = Budget(args.transfer_mib * 1024**2)
    report = {"dataset": "restor/tcd", "revision": REVISION, "official_code_revision": CODE_REVISION,
              "utc": datetime.now(timezone.utc).isoformat(), "samples": [], "status": "INCOMPLETE",
              "limits": vars(args) | {"output": "external cache"}}
    try:
        info = json.loads(budget.get(f"https://huggingface.co/api/datasets/restor/tcd/revision/{REVISION}?blobs=true"))
        shard = next(s for s in info["siblings"] if s["rfilename"] == "data/train-00000-of-00007.parquet")
        remote = RangeFile(ROOT + "/" + shard["rfilename"], shard["size"], budget)
        parquet = pq.ParquetFile(remote)
        metadata_columns = ["image_id", "oam_id", "validation_fold", "license", "segments", "coco_annotations", "meta", "height", "width"]
        rows = []
        for batch in parquet.iter_batches(batch_size=1, columns=metadata_columns):
            rows.extend(batch.to_pylist())
            if len(rows) >= args.scan_limit:
                break
        report["scanned"] = len(rows)
        selected = choose(rows, 2)
        stored = 0
        cached_group, cached_pairs = None, None
        for row in selected:
            # Locate selected row in its original row group. Read only that group's image columns.
            index = next(i for i, r in enumerate(rows) if r["image_id"] == row["image_id"])
            offset = 0
            for group in range(parquet.num_row_groups):
                length = parquet.metadata.row_group(group).num_rows
                if index < offset + length:
                    if cached_group != group:
                        cached_pairs = parquet.read_row_group(group, columns=["image", "annotation"])
                        cached_group = group
                    pair = cached_pairs.slice(index-offset, 1).to_pylist()[0]
                    break
                offset += length
            image_bytes, annotation_bytes = pair["image"]["bytes"], pair["annotation"]["bytes"]
            required = len(image_bytes) + len(annotation_bytes) + len(row["segments"].encode()) + len(row["coco_annotations"].encode()) + row["height"] * row["width"] + 65536
            if stored + required > args.storage_mib * 1024**2:
                raise ValueError("Storage budget exceeded")
            folder = output / str(row["image_id"])
            hashes, qa = save_pair(folder, row, image_bytes, annotation_bytes)
            stored += sum(p.stat().st_size for p in folder.iterdir())
            source_url = row["meta"].replace("http://", "https://", 1)
            if urlparse(source_url).hostname != "oin-hotosm.s3.amazonaws.com":
                raise ValueError("Unexpected OAM metadata host")
            metadata_error = None
            try:
                meta_bytes = budget.get(source_url)
            except requests.RequestException as error:
                meta_bytes = b""
                metadata_error = type(error).__name__ + " (source metadata unavailable)"
            if stored + len(meta_bytes) > args.storage_mib * 1024**2:
                raise ValueError("Metadata exceeds storage budget")
            (folder / "oam-meta.json").write_bytes(meta_bytes)
            stored += len(meta_bytes)
            # Raw metadata may include personal contacts. Publish only reviewed rights fields separately.
            report["samples"].append({k: row[k] for k in ("image_id", "oam_id", "validation_fold", "license", "split", "height", "width")} | {
                "hashes": hashes, "qa": qa, "source_metadata_url": source_url,
                "source_metadata_sha256": digest(meta_bytes) if meta_bytes else None,
                "metadata_error": metadata_error, "training_eligible": False,
                "rights_status": "PENDING_MANUAL_REVIEW",
                "source_provider": "UNVERIFIED", "source_license": "UNVERIFIED",
                "crowd_status": "PENDING_SEMANTICS" if qa["crowd_ids"] else "ABSENT", "background_status": "UNANNOTATED_ZERO_PENDING",
                "annotation_terms": "CC BY 4.0 per official docs/dataset.md at recorded code revision",
                "shard": shard["rfilename"], "row_index": index})
        report["status"] = "PIXEL_QA_COMPLETE_RIGHTS_PENDING" if len(report["samples"]) == 2 else "INSUFFICIENT_SAMPLES"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["transfer_payload_bytes"] = budget.used
    report["stored_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PIXEL_QA_COMPLETE_RIGHTS_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
