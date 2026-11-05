# M1-5A dataset inventory and bounded expansion plan

## Method and evidence boundaries

Revision d97d4da0ebbb6e249ae95ac5e19656babd972eb2 of restor/tcd is the sole dataset. Reference official TCD code revision is 05dab754a3fd15b83d6b7606de7310155f4e2df2. The official card reports 4,169 train and 439 test records. The term official train includes our internal Train/Val folds. Official test is held out from acquisition candidates, training and tuning.

The inventory reads only image_id, oam_id, license, validation_fold, height, width, crs, bounds, biome and meta columns through bounded HTTP Range requests. It does not read image or annotation columns. Test metadata is used solely to verify group/footprint separation. No test pixels or labels are read. The 128 MiB response-payload ceiling includes metadata requests and selected STAC records. Headers/TLS overhead are excluded; the existing reader can exceed the counter ceiling by one rejected 64 KiB chunk before aborting.

Image dimensions/CRS/bounds in Parquet are row metadata. Original format and embedded GeoTIFF affine, valid masks, raster orientation and actual alignment remain unverified for new candidates. The card describes 2048 x 2048 RGB GeoTIFF and COCO-style panoptic labels. The previously audited two originals are TIFF with PNG annotations; extrapolating that observed format to every record requires original-byte checks.

## Partition rules

Official train validation_fold 0 maps to Val; folds 1–4 to Train. Official test maps to Holdout, regardless of fold value. Other train fold values are held. Any duplicate image_id or oam_id crossing folds/roles/official splits stops selection. Repeated oam_id within the same fold is expected source tiling and is counted.

Retain 661 in Train and 1972 in Val, preserving their existing M1-2A/B rights evidence, input hashes and downstream manifests. Candidate expansion excludes both complete source groups. Each selected new source contributes at most one tile.

All candidate metadata footprints are checked against every opposite-role footprint, including Holdout. A positive-area cross-role bounding-box intersection holds that candidate. This is a conservative metadata-only screen. Float32 bounds, unknown orthorectification and separate oam IDs for neighboring/duplicated surveys limit its precision. Adjacency and short-distance proximity require a further geographic review in M1-5B. Avoid claiming universal no-leakage from oam_id alone. Same-role adjacent tiles are not selected here because each source is used only once, though independent source IDs may still describe nearby imagery.

## Predeclared deterministic selection rule

Select up to eight Train and four Val records in that order. Eligibility requires complete required fields, row license exactly CC-BY 4.0, valid official train fold and transformable metadata extent. Exclude existing/selected oam_id and opposite-role footprint overlaps. Among remaining records, prefer an unused 10-degree lon/lat cell, then an unused biome code, then the lowest image_id. Selected IDs are removed before the next iteration. This spreads coarse geography and source groups without claiming a representative global sample.

Canopy coverage, optical quality and embedded local GSD cannot be established from the selected metadata columns. They are deliberately unscored in this first selection. M1-5B should obtain only proposed candidates under explicit per-file and total budgets, verify source/pixel quality, measure canopy fractions and reject/replace unsuitable candidates through a separately logged deterministic review. No numeric coverage values are invented.

## Rights and provenance tiers

The official dataset card and docs/dataset.md declare restor/tcd as CC BY 4.0-only, annotations CC BY 4.0 and imagery under declared OAM source licenses. Each selected row's license must agree. Original-source STAC is queried only for selected provisional candidates and keyed by exact oam_id. An exact item with CC-BY-4.0 supplies corroborating source evidence; conflicting/missing rights stay HOLD. Provider presence is recorded without publishing names, descriptions, emails or contacts. Raw STAC snapshots remain external and their hashes and canonical URLs are published. Attribution completeness still requires manual review of the externally retained original source record.

All new candidates retain training_eligible=false until raw image/label/grid/validity and per-source rights QA pass. Source-declared/provisional classification does not authorize unconditional training. Existing conditional acceptance for 661/1972 is retained. Organizational commercial production rights review is a separate gate.

## GSD and label policy carried forward

Use the existing 0.50m ground-scale-aware, extent-preserving EPSG:3395 method after verifying each original TIFF. A metadata-only estimate derives map intervals from bounds/dimensions, then measures center-local WGS84 east/north intervals. These are provisional because HF bounds use float32 and original affine/CRS semantics have not yet been checked. Do not use fixed factor five. No reprojection is justified solely by this inventory.

For new originals, verify embedded geometry, source footprint, source/label dimensions/orientation, nodata/valid mask and local scale before preparing output. Use one shared output grid for RGB, canopy_fraction, thresholded canopy_mask and valid_fraction. Categories remain Canopy 1 and Tree 2, panoptic ID=R+256G+65536B; category-validated known crowd pixels join positives under the reviewed semantic policy. ID 0 is annotation-convention background and is separate from spatial invalidity. Unknown nonzero IDs/categories fail. Actual ID/category distributions, crowd presence and ecological annotation completeness remain unknown for every new candidate until raw QA.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/m1_5a_inventory.py --cache '<fresh-external-metadata-cache>'
.\.venv\Scripts\python.exe scripts/m1_5a_inventory.py --cache '<same-cache>' --offline
.\.venv\Scripts\python.exe -m pytest -q
```

Existing pyarrow, requests, NumPy and pyproj suffice. External cache stores compact rows, acquisition provenance, per-row disposition reasons and raw selected STAC snapshots. Git receives aggregate results, candidate records stripped of personal attribution information, code, synthetic tests and this plan. Offline replay reads cached metadata/evidence, performs no network requests and repeats deterministic candidate selection.

References are the [fixed dataset card](https://huggingface.co/datasets/restor/tcd/blob/d97d4da0ebbb6e249ae95ac5e19656babd972eb2/README.md), [official split/license tests](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/tests/unit/test_oam_tcd_dataset.py), [dataset documentation](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/docs/dataset.md) and [official export/category code](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/tools/generate_dataset.py). The first two were re-read successfully via direct HTTP during this audit after browser fetch failures.

## Actual metadata results, 2026-11-05

Full revision inventory contains 4,608 records, 541 unique oam_id values. All row licenses are CC-BY 4.0, row dimensions 2048 x 2048 and metadata CRS EPSG:3395. Official train has 4,169 rows from 486 sources. Holdout has 439 rows from 55 sources. Internal Train folds 1–4 total 3,341 rows / 384 sources, Val fold0 has 828 rows / 102 sources. Fold counts are 0=828, 1=945, 2=792, 3=927, 4=677 and -1=439. Repeated oam_id occurs in 446 source groups, with no group crossing folds or roles. Existing 661/1972 identity, license, fold and role matched the prior manifest. Their original hashes and accepted rights records remain untouched.

The [actual inventory JSON](M1_5A_INVENTORY_RESULT.json) records aggregate distributions, exact shard references and candidate decisions. The external rows cache is 3,236,857 bytes with SHA-256 f83ad4c27154d62e485d59b4abc5148c74fc5760c090530663ee927ca1566c27. Successful acquisition received 924,991 metadata payload bytes and 963,094 total bytes including twelve selected STAC items. Publisher full-shard hashes are recorded as provenance but were not independently recomputed, since full shards were not downloaded.

Two earlier metadata-only attempts were stopped for excessive latency before any completed row cache or result was persisted. Their intermediate transfer counts are NOT_RECORDED; each had a 128 MiB ceiling. Additional small documentation/footer diagnostics were outside the final successful acquisition counter. Thus 963,094 bytes is the successful run amount, not total session network usage. Final code reuses HTTP connections and resolved signed redirects, caches CRS transformers and avoids repeated opposite-role list construction. No image/annotation column was requested in any attempt. Parquet footer range reads may include adjacent encoded bytes, but no image column was decoded or saved.

All 4,608 metadata bounds could be transformed. Center latitude spans -42.5591 to 65.7584 degrees, longitude -175.3038 to 177.5352 degrees. Center-local metadata-derived intervals span approximately 0.04102 to 0.10010 m/px, median 0.09266m. These are estimates from row bounds, with quantization and embedded-original verification pending. Coarse cell counts are retained for geographic inspection.

Candidate reason counts include 439 Holdout exclusions, 81 rows blocked by existing/selected source groups, 55 rows held for cross-role footprint overlap, 12 selected for source investigation and 4,021 other rows outside the small candidate budget. The overlap result concerns different oam IDs and is distinct from the zero source-group conflicts. Those 55 rows are excluded from this expansion proposal. Nearest opposite-role center distances for the selected set are recorded; the smallest is about 19.19km. Center distance is an approximate proximity screen and does not prove survey independence.

## M1-5B proposed acquisition list

No new candidate is approved for training. Initial source-declared selection produced eight Train and four Val IDs. Source checks then held image1 because the exact STAC item has no license field. This is missing evidence, not proof of a restrictive license. The publisher and row declare CC BY 4.0, while original-source terms remain unresolved. No automatic replacement was made. All other selected STAC records declare CC-BY-4.0 and include provider records, with attribution completeness still pending manual inspection.

| ID | Role | Lon / lat | Estimated ground m/px E/N | M1-5B disposition |
| --- | --- | --- | --- | --- |
| 1 | train | -99.16451, 19.35060 | 0.09448 / 0.09436 | HOLD |
| 2 | train | -70.32295, -27.35440 | 0.08875 / 0.08886 | PROVISIONAL_ACQUIRE_FOR_QA |
| 3 | train | 28.84598, -15.95095 | 0.09615 / 0.09621 | PROVISIONAL_ACQUIRE_FOR_QA |
| 4 | train | -99.84316, 32.65683 | 0.08436 / 0.08425 | PROVISIONAL_ACQUIRE_FOR_QA |
| 5 | train | -58.74424, -9.61902 | 0.09846 / 0.09864 | PROVISIONAL_ACQUIRE_FOR_QA |
| 6 | train | 120.20357, 11.99773 | 0.09793 / 0.09787 | PROVISIONAL_ACQUIRE_FOR_QA |
| 11 | train | -65.61923, -17.62257 | 0.09520 / 0.09531 | PROVISIONAL_ACQUIRE_FOR_QA |
| 17 | train | 174.92084, -41.19714 | 0.07580 / 0.07524 | PROVISIONAL_ACQUIRE_FOR_QA |
| 9 | val | -81.56842, 41.58456 | 0.07498 / 0.07498 | PROVISIONAL_ACQUIRE_FOR_QA |
| 576 | val | -119.26526, 52.77165 | 0.06069 / 0.06069 | PROVISIONAL_ACQUIRE_FOR_QA |
| 1401 | val | -79.91618, -3.28815 | 0.09945 / 0.09984 | PROVISIONAL_ACQUIRE_FOR_QA |
| 4235 | val | 83.90990, 28.13267 | 0.08834 / 0.08823 | PROVISIONAL_ACQUIRE_FOR_QA |

Recommend seven new Train and four Val pairs for bounded raw QA in M1-5B, with image1 held. Each has a distinct new oam_id, verified-source-license metadata and no opposite-role bbox overlap. Their specific source IDs, shard/row locations, licenses, source URLs and snapshot hashes are in the inventory JSON.

Before requesting bytes, inspect selected row-group image/annotation compressed sizes and set a fixed total response/storage budget. Group candidate reads by shard/row-group and reuse cached columns. Stop rather than auto-increase limits if too much unrelated compressed data would be fetched. A provisional M1-5B ceiling of 256 MiB transfer and 128 MiB saved artifacts is proposed for review, with fewer pairs if required. Exact bytes needed are unmeasured here.

For each acquired pair recheck raw hashes, source STAC snapshot identity/attribution, ID-category mapping, crowd, zero background, validity and RGB alignment. Preserve new sources/roles in a manifest. Measure actual canopy fraction and note sparse/dense/ambiguous cases; the current metadata selection does not guarantee coverage diversity. Verify original affine and local ground GSD, then apply the reviewed approximately 0.50m method. Reject or hold failures before any optimizer use. Existing two accepted samples retain their current roles.

## Verification and unresolved items

Final offline tests passed with 60 tests in 11.03 seconds and 19 existing Affine warnings; pip check succeeded. Tests cover source-group duplication, fold/role/Holdout conflicts, unknown license/missing fields, deterministic selection under reordered input, spatial-overlap holds and existing sample identity/role preservation. The candidate IDs and source status were reproduced offline using the completed cache.

Whole-corpus original-source rights/attribution, raw images, label pixels, actual embedded geometry, nodata and optical quality remain unverified beyond the existing two samples. New candidates remain provisional. The dataset-scale metadata audit and bounded expansion plan are PASS candidates for review, with raw QA/training eligibility deferred to M1-5B.

The full real metadata cache was also reordered in reverse and candidate selection was rerun offline. Selected image IDs matched exactly. Published per-row disposition cache SHA-256 ties all exclusion reasons to the external audit artifact.
