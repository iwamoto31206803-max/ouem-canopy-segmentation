# OAM-TCD technical audit — research notes 05: source-level split tests and coordinate consistency

Status: **read-only targeted audit; published data have not been fully downloaded or revalidated**. 2026-10-31. Upstream GitHub source at default-branch snapshot, not yet pinned; do not conflate the existence of tests with passing those tests on the current dataset.

## Source: upstream dataset sanity tests
[tests/unit/test_oam_tcd_dataset.py](https://github.com/Restor-Foundation/tcd/blob/main/tests/unit/test_oam_tcd_dataset.py) is a first-party check that:
- loads `restor/tcd`, `restor/tcd-nc`, `restor/tcd-sa`;
- asserts that `oam_id` does not overlap among any distinct training `validation_fold` values 0..4;
- asserts that each training fold's `oam_id` values do not overlap with the `test` split;
- asserts every split of each license-specific repository uses exactly one expected license string.
The code makes no assertion regarding the **current runtime outcome** without executing the tests. It also does not independently validate geographic coordinates, masks, fold representativeness or downstream modeling quality.

## Coordinate finding requiring follow-up
The [public CC BY dataset viewer](https://huggingface.co/datasets/restor/tcd) includes source projected `bounds`, `crs = EPSG:3395`, and `lat`/`lon` columns. On at least two viewed records, reprojecting the center of `bounds` EPSG:3395 → EPSG:4326 (pyproj Transformer, always_xy=True) gives the following:

| image_id | calculated (longitude, latitude), degrees | viewer fields approx. |
| --- | --- | --- |
| 661 | (31.41786, 50.03229) | lat=31.41694, lon=50.0317 |
| 2039 | (-2.00195, 5.29981) | lat=-2.00287, lon=5.298892 |
| 1972 | (39.00465, 55.81684) | lat=39.003735, lon=55.816322 |

The viewer `lat` values resemble **longitude**, and `lon` resemble **latitude**, with small differences reflecting exact point definition or projection geometry. **Suspected field-name reversal**, not confirmed universal error. Do not silently swap all records until sampling many records across hemispheres and inspecting upstream data creation. The `bounds`+CRS may provide a more reliable derived coordinate after checks. For spatial stratification, treat raw `lat` and `lon` as unverified.

## Research steps completed
- Inspected public rows and metadata only; no bulk download and no official holdout used to tune or train.
- Tried Hugging Face API via read-only web access; endpoints could not be retrieved through available interface, so exact shard counts, revision commit and pixel-level sample validation remain **NOT VERIFIED**.
- Identified first-party split and license unit tests as the canonical logic to reproduce, not an excuse to skip our own grouped integrity checks.

## Next read-only validation gates
1. Pin dataset and code revisions; collect exact CC BY train/test counts and source `oam_id` summaries (metadata only).
2. Verify `lat`, `lon`, projected `bounds`, point definition and potential column reversal on representative global records; report prevalence and whether metadata authoring or viewer causes it.
3. Obtain a bounded set of real **training-only** image/annotation pairs and inspect RGB segment-ID mapping, crowd and void encoding, affine transforms and source rights.
4. Reproduce upstream grouping and license tests at pinned revisions, then develop independent property-based checks.
5. Record exceptions and keep official test untouched for model tuning.

**No Phase 1 implementation or architecture recommendation is approved by this note.**
