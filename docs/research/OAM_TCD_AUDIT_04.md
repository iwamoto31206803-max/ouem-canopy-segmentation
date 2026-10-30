# OAM-TCD technical audit — research notes 04: CC BY dataset records

Status: evidence from the live Hugging Face dataset viewer, repository file listings, official dataset card, and upstream code; **no whole-dataset scan, decoded mask QA or downloaded dataset**. Date: 2026-10-31. Read-only investigation. Pin commit hash and recheck exact cardinalities before any implementation.

## Primary sources
- [Restor TCD dataset viewer/card](https://huggingface.co/datasets/restor/tcd)
- [Main files](https://huggingface.co/datasets/restor/tcd/tree/main)
- [Parquet files](https://huggingface.co/datasets/restor/tcd/tree/main/data)
- [NeurIPS 2024 supplementary](https://papers.nips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Supplemental-Datasets_and_Benchmarks_Track.pdf)
- [Upstream dataset generator](https://github.com/Restor-Foundation/tcd/blob/main/tools/generate_dataset.py)

## Confirmed from public viewer / card
- `restor/tcd` marked CC BY 4.0, with dataset card stating 4,608 records in the CC BY subset of 5,072 total across all license-specific repositories. Full-all-license aggregate card numbers train 4,406 / test 666 **must not be attributed to this CC BY-only repo**.
- Viewer displays two splits and **train approximately 4.17k**; this UI rounds counts. Exact CC BY-only train and test counts **not verified**.
- `data/` contains one `test-00000-of-00001.parquet` and seven train shards `train-00000-of-00007.parquet` to `train-00006-of-00007.parquet`. UI lists approx 3.55 GB repo size. No full-shard local download performed.
- Viewer shows fields: `image_id`, `image`, `height`, `width`, `annotation`, `oam_id`, `license`, `biome`, `crs`, `bounds`, `validation_fold`, `biome_name`, `lat`, `lon`, `segments`, `meta`, `coco_annotations`.
- Visible records are 2048×2048 and display EPSG:3395. Example record: image_id 661, oam_id `60f0a29989ec130007b828a8`, validation_fold=2, license=`CC-BY 4.0`, categories 1 and 2 in `segments`; this is not proof every record has the same CRS or metadata quality.
- Visible `segments` values are JSON strings containing object `id`, `category_id`, `area`, `bbox`, `iscrowd`. `coco_annotations` includes `image_id` and segmentation content. Both polygon-like and RLE-like `segmentation` structures appear in the preview. Treat `annotation` as RGB-coded instance IDs rather than categorical pixel codes.
- Sample records include `iscrowd=1` with `area=0`, and polygon coordinates extending slightly beyond 2048; **do not call errors or assume these areas are truly empty without decoder/raster comparison**.
- Official dataset card explains `oam_id` grouping across splits/folds to prevent leakage; `validation_fold` in [0,4] for train. Official holdout is not a validation fold and must remain independent.
- Card notes labels were created by several annotation teams with most images annotated by one person, and potential missed/misclassified canopy and spatial/biome sampling bias. Unlabeled pixels are not automatically audited true-negative background.

## Mask conversion implications
- Upstream `tools/generate_dataset.py` optional binary mode transforms RGB annotation pixels to canopy using `np.any(mask != 0, axis=-1)`. This is a source-backed baseline for semantic coverage, not proof all dataset edge cases have acceptable label semantics.
- Preserve original `annotation` bytes, `segments` and `coco_annotations`; independently compare simple nonzero conversion with decoded segment category maps and crowd/void logic.
- Proposed QA: verify annotation RGB↔segment ID mapping, category 1 canopy / 2 tree, unknown IDs, overlap and boundary clipping, void/crowd, fraction canopy, all-zero/all-one masks, source pixel alignment, input bounds and per-record rights.
- Stratify initial inspection beyond random images: tree-dominant, canopy-dominant, mixed, `iscrowd`, sparse/negative, multiple biomes, boundaries.

## Unverified
1. Exact split counts and revision SHA from HF API; endpoint was not accessible in this review, so no invented values.
2. Downloaded original image/annotation pixel samples, segment-ID decoding, per-image hash, source metadata completeness.
3. Full grouping statistics (`oam_id` counts, cross-fold integrity), exact unique source sizes, distribution and label completeness.
4. Whether image `crs`, geospatial bounds, and original 10 cm resampling assumptions are uniform for all records.
5. Attribution format, provider-specific provenance / extra rights requirements beyond the CC BY repo badge.
6. Official panoptic-to-semantic conversion consistency for all corner cases.

## Research decision
No code, dataset sampling or Phase 1 training approved yet. Next step: **bounded, reproducible local inspection of real CC BY rows without reading official holdout for tuning**, using pinned dataset revision, and exporting only metadata+QA summaries (no imagery in Git). Then revisit evaluation design / model selection.
