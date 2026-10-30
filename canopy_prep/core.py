"""Offline, deterministic canopy preprocessing; no upstream schema assumptions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def binary_canopy_mask(labels: np.ndarray, tree_values: set[int]) -> np.ndarray:
    """Accept semantic class IDs, NOT raw panoptic segment IDs."""
    a = np.asarray(labels)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or not tree_values:
        raise ValueError("expected 2D integer semantic labels and explicit tree IDs")
    return np.isin(a, list(tree_values)).astype(np.uint8)

def degrade_pair(rgb: np.ndarray, mask: np.ndarray, factor: int = 5, threshold: float = 0.5):
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 HxWx3")
    if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be aligned binary HxW")
    if not isinstance(factor, int) or factor < 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid factor or threshold")
    h, w = mask.shape
    hh, ww = h // factor * factor, w // factor * factor
    if not hh or not ww:
        raise ValueError("image smaller than factor")
    low = np.floor(rgb[:hh, :ww].reshape(hh//factor, factor, ww//factor, factor, 3).mean((1, 3)) + 0.5).astype(np.uint8)
    frac = mask[:hh, :ww].reshape(hh//factor, factor, ww//factor, factor).mean((1, 3)).astype(np.float32)
    return low, frac, (frac >= threshold).astype(np.uint8)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_pair(rgb, mask, destination: Path, *, source_id: str, oam_id: str, split: str, factor: int = 5):
    if split not in {"train", "validation"} or not source_id or not oam_id:
        raise ValueError("invalid identifiers or split")
    if any(c in source_id for c in ("/", "\\", "..")):
        raise ValueError("unsafe source ID")
    destination = Path(destination)
    folder = destination / split / source_id
    if folder.exists():
        raise FileExistsError(folder)
    low, frac, binary = degrade_pair(rgb, mask, factor)
    folder.mkdir(parents=True)
    paths = {"rgb": folder/"rgb.png", "fraction": folder/"canopy_fraction.npy", "mask": folder/"mask.png"}
    Image.fromarray(low).save(paths["rgb"])
    np.save(paths["fraction"], frac, allow_pickle=False)
    Image.fromarray(binary*255).save(paths["mask"])
    record = {"source_id": source_id, "oam_id": oam_id, "split": split, "factor": factor, "threshold": 0.5, "source_shape": list(np.asarray(mask).shape), "output_shape": list(binary.shape), "files": {k: {"path": p.relative_to(destination).as_posix(), "sha256": sha256(p)} for k,p in paths.items()}}
    (folder/"manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return record

def validate_disjoint(records: list[dict]) -> None:
    seen = {}
    for record in records:
        key, split = record["oam_id"], record["split"]
        if key in seen and seen[key] != split:
            raise ValueError(f"oam_id leakage: {key}")
        seen[key] = split
