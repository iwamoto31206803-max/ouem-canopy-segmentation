# M1-2A bounded raw-pair audit

## Reproduce

Use the existing Windows Python 3.11.9 venv. This work added only `pyarrow==18.1.0`. Existing requests, NumPy and Pillow provide HTTP access and image decoding. `pip check` succeeded.

```powershell
.\.venv\Scripts\python.exe -m pip install pyarrow==18.1.0
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/m1_2a_data_qa.py --output '<fresh-external-cache-directory>' --scan-limit 32 --transfer-mib 128 --storage-mib 32
```

The output must be a fresh directory outside this repository. Selection uses source order in the first train shard, chooses the first valid train and val rows, and rejects observed source-group overlap. Scan is limited to 32 records by default and 64 maximum. Selected pair count is fixed at two. Fold 0 maps to val and folds 1–4 to train. All source objects are revision-pinned. HTTP byte ranges read Parquet columns and reuse the selected row group's image columns. The viewer JPEG previews are unsuitable for panoptic ID verification and are unused.

Transfer budget measures HTTP response payload bytes received by this script, including repeated requests. Protocol headers and TLS overhead are excluded. A rejected chunk can exceed the counter's ceiling by at most one 64 KiB chunk before abort. Range requests larger than the remaining budget are rejected before submission. Saved original and candidate file sizes are checked against a storage budget before writing. The small final JSON report adds overhead beyond that data budget. No shard is cached on disk. Row-group decompression can consume more memory than transfer size.

Original embedded image bytes, annotation bytes, segment/COCO JSON and candidate binary PNG remain outside Git. The published manifest contains reviewed identifiers, aggregate QA and hashes. Source metadata can include personal contacts and stays in the external cache when available. Provider/rights evidence requires manual review before publishing selected fields.

## Observed results, 2026-11-05

Dataset `restor/tcd` revision `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`. Official code reference `05dab754a3fd15b83d6b7606de7310155f4e2df2`. Only the first official train shard supplied sample bytes.

The initial attempt stopped on OAM metadata HTTP 403 after saving the first pair. A minimal correction preserves pixel evidence and marks missing rights as pending, allowing the second pair to be audited. Both attempts received 81,380,037 payload bytes each, approximately 155.2 MiB combined. The successful audit stored 1,801,688 data bytes, plus its report. Both attempts remain externally cached and are bounded separately. The first failed attempt saved 1,024,411 data bytes.

| image_id | fold / assignment | dimensions | category IDs | positive pixels | ID 0 pixels | crowd IDs |
| --- | --- | --- | --- | --- | --- | --- |
| 661 | 2 / train | 2048 x 2048 | 1, 2 | 2,200,120 | 1,994,184 | none |
| 1972 | 0 / val | 2048 x 2048 | 1, 2 | 2,543,750 | 1,650,554 | 106343 |

Both pairs decoded successfully. Dimensions match, every nonzero RGB-coded ID maps to a segment, categories are limited to Canopy 1 and Tree 2, and every listed segment has pixels. Source `oam_id` values differ. The [manifest](M1_2A_SAMPLE_MANIFEST.json) records exact original-byte hashes and reference information.

Binary candidate masks merge category 1 and 2 after ID lookup. ID 0 is recorded as unannotated background candidate. Crowd pixels retain their category in the diagnostic candidate and their IDs are recorded. The candidate PNG is a diagnostic artifact. Training eligibility remains false for both samples pending background/validity policy, crowd semantics and source rights verification. This audit does not establish image registration from dimensions alone. Independent polygon/RLE rasterization and visual overlay comparison remain pending.

Both original OAM metadata endpoints returned HTTP 403. Row licenses are `CC-BY 4.0`, while original imagery provider/license fields remain UNVERIFIED. Annotation terms are supported by the official dataset documentation at the fixed code revision. The resulting status is `PIXEL_QA_COMPLETE_RIGHTS_PENDING`, with two pixel-audited pairs and zero fully eligible training samples.

Offline tests cover RGB-ID reconstruction, category union, unknown IDs/categories, dimensions, uniform masks, deterministic selection, source-group overlap, saved-file hashes and transfer preflight limits. Synthetic tests and real-pair evidence are tracked separately.

## Handoff

Resolve original imagery attribution/license evidence through an authoritative source, confirm zero/void/valid-area semantics and crowd policy, and inspect overlays before releasing these samples for training. Broader fold/group validation awaits a metadata-only audit. GSD selection and resolution transforms follow the separate GSI audit.

References are the [fixed dataset card](https://huggingface.co/datasets/restor/tcd/blob/d97d4da0ebbb6e249ae95ac5e19656babd972eb2/README.md), [official dataset documentation](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/docs/dataset.md) and [official converter](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/tools/generate_dataset.py).

Final offline suite completed with 17 passed in 8.84 seconds, including existing M1-1 tests. An independent reread of all published real-sample file hashes matched the external cache.
