# OAM-TCD technical audit — research notes 06: bounded real-record access attempt

Status: 2026-10-31, **accessibility test and validation protocol, not pixel-level acceptance**. Research only: no downloaded training imagery, no inference, no change to project code. Official test/holdout not accessed for tuning.

## Attempted access and actual outcomes

1. Opened [restor/tcd HF dataset viewer](https://huggingface.co/datasets/restor/tcd), which exposes **preview metadata for actual train rows**, including image/annotation thumbnails, ids, `oam_id`, license, `validation_fold`, image dimensions and location fields. These **viewer-rendered metadata** are inspectable and do not require full Parquet acquisition.
2. Opened [HF data file listing](https://huggingface.co/datasets/restor/tcd/tree/main/data): seven train Parquet shards and one test shard, ~3.55 GB total. Full download was not attempted.
3. Tried `https://datasets-server.huggingface.co/splits?dataset=restor%2Ftcd` and `https://huggingface.co/api/datasets/restor/tcd` via available text web access. Neither endpoint produced accessible JSON here.
4. Followed the first train preview image asset from the viewer: target is a dataset-server image endpoint, but this interface returned **unsupported binary/octet-stream**, not original image bytes. This does not establish a dataset failure.
5. Attempted read-only Python HTTP request from the execution container to the dataset-server: DNS resolution blocked, so no record image, annotation, dataset script or Parquet bytes were downloaded.
6. Official test in [tests/unit/test_oam_tcd_dataset.py](https://github.com/Restor-Foundation/tcd/blob/main/tests/unit/test_oam_tcd_dataset.py) verifies between-fold `oam_id` disjointness and split license consistency by loading all three license-specific repos, but we did **not** execute this test.

## Evidence grades
- **Observed:** webpage data-card declarations, file listing, row-preview fields and sample metadata, first-party source implementation.
- **Not observed:** raw RGB and annotation pixels, matched `segments` ids per pixel, exact split sizes, full grouped fold counts, current dataset revision hash (a URL contains a revision-like hash, but it is not independently confirmed as canonical).
- **Not inferred:** complete source data correctness; a viewer preview is not real-image QA.

## Concrete bounded verification protocol for Windows CPU (design only; not executable script)
1. Resolve and record full dataset commit SHA, config, available splits, row counts, explicit CC BY license string; keep source revision immutable.
2. Obtain only a *small, bounded training subset* with `image`, `annotation`, `segments`, `coco_annotations`, `image_id`, `oam_id`, fold, CRS/bounds, license. Avoid all test/holdout downloads.
3. Verify identical 2048×2048 source dimensions; bit-depth and image modes; calculate raw SHA-256 before any transforms.
4. Decode panoptic RGB segment IDs and check against `segments` IDs and categories, including ids not in metadata, background, crowd and polygons/RLE; compare to official nonzero-RGB semantic-mask behavior.
5. Compare source geospatial bounds with reconstructed lon/lat (EPSG:3395→EPSG:4326 with always-xy semantics); flag swapped `lat`/`lon`, coordinate outliers and source pixel-size variation. **Do not silently correct original metadata.**
6. Audit `oam_id` disjointness for all official training folds, using metadata only; verify source groups cannot cross train/validation or independent test.
7. Produce machine-readable per-record QA ledger plus human-reviewed image/mask overlays and exception log, without committing bulky data to GitHub.
8. Require explicit PASS of pixel and grouping QA before prescribing any resampling, model initialization or training procedure.

## Priority open question
The original 10 cm labels may have annotation ambiguity and incomplete canopy boundaries. Treat semantic binary conversion as a **source-backed candidate**, not a validated universal truth. Define separate evidence levels for code correctness versus label truth.

## Outcome
**Blocked on direct binary access in this environment.** No misleading real-data PASS. Next meaningful action: obtain a few training-only sample records in an environment with Hugging Face access (e.g. GEEKOM-A6), or inspect another provider-supported unmodified binary route; bring actual QA outputs back for design review. No need to download the entire dataset or implement training first.
