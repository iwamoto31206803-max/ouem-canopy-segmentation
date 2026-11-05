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

## M1-5B implementation and pilot audit (2026-11-05)

**Code implementation: PASS candidate. Real-data pilot: HOLD / NOT PASSED.** The new [single CLI](../../scripts/m1_5b_pilot.py) separates `plan`/`acquire` network operations from offline `prepare`/`verify`. [Sanitized measured results](M1_5B_PILOT_RESULT.json) contain hashes and geometry, not images or attribution contacts. No existing source, accepted pair, training/GPU script or formal design was changed. This section supplements the historical M1-5A selection above.

### Reuse and gates

The CLI reuses M1-5A role/split/overlap checks and pinned metadata; M1-2A `labels`/`save_pair`; M1-2B overlay/RLE checks; M1-2D `intervals`, exact area weights, output grid and `binary`. The old two-ID `audit`/`prepare`/`load_patch` wrappers stay unchanged: they intentionally accept only 661/1972, so new pairs are **not automatically connected to training**. No new dependency or venv change was needed: requests, pyarrow, NumPy, Pillow, rasterio and pyproj were already available. Python 3.11.9 on Windows was used; resolved versions are in the result JSON. A company PC can use the existing Windows CPU setup and install only missing data-audit packages after reviewing its environment.

Only explicit IDs from the eleven proposed candidates are permitted. Image 1, existing IDs and Holdout IDs are rejected. The inventory `rows.json` must match the committed M1-5A SHA-256; acquisition revision/shard provenance and existing identities/folds/licenses are checked before network work. Missing cache is a stop, not permission to switch revision. Transfer/copy the completed M1-5A cache for offline use. If unavailable, its historical `m1_5a_inventory.py` reproduction commands above regenerate fixed-revision **metadata only** under their separate 128MiB budget; review that additional transfer first. M1-5B does not regenerate it automatically.

Across all cached opposite-role metadata (including Holdout metadata only), gates reject common oam_id, an identical original metadata object key even across the migrated OAM bucket names, footprint intersection, and a conservative center-distance-minus-footprint-radii separation below 1km. The 1km threshold was declared before pixel acquisition as a proximity hold, not proof of survey independence. Exact identity aliases absent from metadata cannot be universally ruled out. No Holdout image/annotation column is requested. The existing Train 661 and Val 1972 roles and all eight accepted output hashes were rechecked unchanged.

Rights gates require row CC-BY 4.0, exact STAC item ID and snapshot hash, source CC-BY-4.0, a named licensor, producer identity and matching original metadata object key. STAC provider/description data stays external with each raw pair. Annotation terms refer to the fixed official dataset documentation; no Restor noncommercial weights are used. These are corroborated source declarations with attribution retained, not final commercial legal approval. Manual attribution review remains necessary before training eligibility.

### Budget and actual candidates

Planning reads publisher metadata and exact Parquet footers first, then calculates required column ranges per target row group. Acquisition prefetches only those ranges, uses cached validated metadata without implicit image reads, extracts selected records and evicts that group's newly created column-cache files. Footer cache, original hashes and request accounting remain. Unselected records in those compressed column chunks are unavoidable row-group transfer overhead; only selected RGB/annotation pairs are decoded and saved.

Limits are 256MiB response payload and 128MiB newly saved cache/products. `network.json` persists consumed bytes, including partial failures; response headers/TLS overhead are excluded. HTTP Range, Content-Range, content length and identity encoding must agree. There are no automatic retries or limit increases. Planning prioritizes 2 and 9, then remaining IDs ascending, to retain one Train and one Val before a second Train. Storage estimates reserve raw RGB/annotation allowance (6 bytes/source pixel plus 1MiB metadata per pair) plus the largest live compressed group; exact pre-write guards still apply. Estimates cannot guarantee arbitrary encoded metadata sizes. A 256KiB report reserve is kept. Do not restart failed acquisition blindly in a new cache: preserve counters and inspect partial artifacts.

| Candidate | Shard / zero-based row group / row in group | Compressed selected columns | Actual disposition |
| --- | --- | ---: | --- |
| Train 2 | train-00000 / 2 / 31 (shard row 231) | 78,284,131 bytes | Acquired; HOLD unresolved black-border validity |
| Train 3 | train-00003 / 1 / 60 (shard row 160) | 78,838,830 bytes | HOLD storage estimate; footer only, no image/annotation columns |
| Val 9 | train-00000 / 5 / 3 (shard row 503) | 74,218,651 bytes | Acquired; HOLD unresolved black-border validity |

All three would require 231,341,612 compressed column bytes, within transfer but over the conservative live-storage estimate. Two groups (100 + 96 encoded records) were fetched to extract just 2 pairs; the other 194 pairs were not decoded as RGB/PNG or saved as individual originals. Actual HTTP payload was **152,548,527 bytes (145.48MiB)**, including 45,745 bytes of API/footer work. Acquisition took 68.32s. Final newly saved pilot cache plus both fresh audit directories totalled **1,935,980 bytes (1.85MiB)** at the recorded observation. Peak disk usage was not separately measured; per-write guards bounded the temporary compressed cache. There was no all-corpus image download.

### Actual source and visual findings

Both originals are 2048 square RGB TIFF, annotations PNG, unrotated EPSG:3395 with projected ~0.10m spacing. Embedded bounds agree with float32 metadata rounding uncertainty; TIFF/PIL pixels agree; annotations have matching dimensions/orientation. IDs/categories are consistent, no crowd was present in these two records, and category 1/2 merging preserves the complete-background mask for 2. Known crowd remains positive by the accepted policy; future crowd RLE must match the panoptic pixels. Unknown IDs/categories/flags fail.

| ID | Source local ground m/px E/N | Diagnostic proposed grid | Diagnostic mask coverage | Validity finding |
| --- | --- | --- | --- | --- |
| 2 | 0.088881 / 0.088881 | 364 square; ~0.500078m | 0%; all annotation background | 1,260,884 exactly black RGB pixels, including 3,391 border pixels |
| 9 | 0.074908 / 0.074908 | 307 square; ~0.499714m | 12.0447% binary positive; mean fraction 12.0566% | 17,102 exactly black RGB pixels, including 354 border pixels |

The source/overlay previews were actually viewed. Image 2 depicts dry terrain/buildings with a large sharp black wedge and an empty annotation. The empty mask is preserved; it does not establish ecological completeness or meaningful positive-mask alignment. Image 9 depicts shoreline, water, trees and buildings; the magenta mask broadly follows visible crowns, with some visible vegetation unlabelled. A small sharp black corner is present. These are visual diagnostics, not accuracy evaluation.

Both TIFFs declare no nodata and GDAL marks all pixels valid. That does **not** establish coverage of the black wedges. The CLI therefore triggers `HOLD_UNMARKED_BLACK_BORDER_VALIDITY` when exact-black RGB occurs on a border. This is a conservative review trigger, **not** a rule that every black pixel is nodata. No footprint correction, crop, ignore label, black-to-background conversion or all-one valid_fraction is emitted for unresolved sources. Invalid pixel count is `null` (unknown), not zero. Both samples remain training-ineligible. No substitute candidate was fetched after discovering the issue.

Proposed output grids and area fractions were computed diagnostically, with source bounds preserved. **No real training GeoTIFF products or output product hashes were created**, so real saved-product readback/determinism are NOT_VERIFIED. Offline verification repeated original hashes, geometry/labels and both HOLD reasons. Resolve actual source coverage/validity evidence and review the appropriate crop/validity policy separately before releasing these records. Do not simply mark the black pixels invalid by color or accept them as negatives. No full-data training is unblocked by this PR.

### Standalone Windows commands

Use PowerShell at the pinned repository checkout and its existing venv. Replace placeholders with external paths. The completed inventory cache must contain `rows.json`, `acquisition.json` and selected `*-stac.json` snapshots. No Codex, ChatGPT, API key or GitHub token is needed. Only the first two stages need internet.

```powershell
$Meta = '<existing-M1-5A-metadata-cache>'
$Cache = '<fresh-external-M1-5B-cache>'
$Out = '<fresh-external-M1-5B-pair-directory>'
.\.venv\Scripts\python.exe scripts/m1_5b_pilot.py plan --inventory-cache $Meta --cache $Cache --ids 2 3 9 --transfer-mib 256 --storage-mib 128
# Inspect plan.json; never continue after failure or auto-raise budgets.
.\.venv\Scripts\python.exe scripts/m1_5b_pilot.py acquire --inventory-cache $Meta --cache $Cache --ids 2 3 9 --transfer-mib 256 --storage-mib 128
.\.venv\Scripts\python.exe scripts/m1_5b_pilot.py prepare --inventory-cache $Meta --cache $Cache --output $Out --ids 2 3 9 --storage-mib 128
# Fully offline, read-only reinspection; emits JSON to stdout:
.\.venv\Scripts\python.exe scripts/m1_5b_pilot.py verify --inventory-cache $Meta --cache $Cache --output $Out --ids 2 3 9
.\.venv\Scripts\python.exe -m pytest -q
```

Check `$LASTEXITCODE` and JSON status after each stage. Exit 0 means that stage completed as specified (raw acquisition still awaits QA); exit 1 means failure; exit 3 means eligibility remains HOLD. `prepare` reserves a fresh directory; `verify` reads the existing one. `acquire` requires a matching plan/budget and a new `raw/` subdirectory; repeated acquisition cannot overwrite originals. Errors and partial acquisitions are not accepted data. Copy the external cache and output manifest together for offline verification on another PC; do not edit hashes to bypass a mismatch.

For a clean source, `prepare` writes shared-grid `rgb.tif`, `canopy_fraction.tif`, `canopy_mask.tif`, `valid_fraction.tif` under `train/<id>` or `val/<id>`, and `manifest.json`. RGB uses exact overlap-area averaging with fixed floor(x+0.5) uint8 rounding; coverage uses identical float64 weights then float32, with mask >=0.5. Uniform masks are retained. Nodata/unknown validity is held, with no invented ignore policy. The saved pixels/grid are read back and hashed.

Clean sources still require an external review JSON passed with `--reviews <file>` to become training-eligible. Keys are image IDs as strings; each entry supplies the actual `preview_sha256`, `stac_sha256`, and `alignment`, `quality`, `attribution` verdicts. Only explicit `PASS` on all three, tied to the exact artifacts, permits eligibility. Record review notes. Review the actual preview and complete external attribution before entering those verdicts. A review file cannot override the black-border validity hold. Missing review yields generated-but-not-approved products.

### Verification and next step

Regression: **102 passed**, including 24 new offline tests, with 75 Affine-related deprecation warnings. Tests cover CLI bounds/IDs, source license/attribution holds, transfer/storage budgets, Range/cache/short reads, protected paths, category/ID rejection, source geometry, crop-free shared grids, 0.5 threshold, uniform labels, split/Holdout/original-object/proximity holds, black-border hold, SHA checks, deterministic synthetic conversion, actual saved-file readback and repeated output hashes. Tests forbid network and do not acquire real data. Real source inspection was a separate bounded execution.

Next action is review of this validity finding and source coverage evidence. The code's valid-source path is tested, but **real-data pilot PASS requires at least one Train and one Val to finish rights, label/validity QA, conversion and saved-product verification**. That condition was not met here. Existing 661/1972 remain the only accepted training pair set; their loader/training behavior and all Satlas/GPU code are unchanged.

## M1-5B-2 black-border cause investigation (2026-11-05)

**Recommendation: use option A now—exclude 2/9 from the next expansion pilot and plan the next approved candidates. Do not implement validity masking or change training/loss at this point.** The dominant wedges are highly likely coverage/fill artifacts, but that cause is not proven. Small interior black pixels can be ordinary dark/shadow pixels. Keep the current pilot HOLD; do not infer dataset-wide occurrence rates from these two purposively selected records.

This is an investigation, not a new preprocessing method. [One offline diagnostic script](../../scripts/m1_5b2_border_audit.py) and extensions to the existing pilot tests reproduce the measurements. [Sanitized result JSON](M1_5B2_BORDER_RESULT.json) records raw hashes, component counts and metadata; all diagnostic images remain external. Existing M1-2D/M1-4/M1-6 and the M1-5B production CLI are unchanged. No new OAM-TCD imagery, upstream OAM raster, checkpoint, training pair, package or GPU operation was acquired/performed. Public documentation was read online.

### Measurements and actual visual inspection

Components use exact decoded RGB=(0,0,0), default 4-neighbor connectivity. Bounding boxes are pixel-edge [left,top,right,bottom), origin top-left. An independent 8-neighbor run and max-channel <=5/<=10 sensitivity checks expose connectivity/compression sensitivity. These thresholds are **diagnostics only**, not NoData/validity definitions. A row-run union/find avoids new scipy/OpenCV dependencies.

| Measurement | Train 2 | Val 9 |
| --- | ---: | ---: |
| Exact black pixels (reproduces PR #25) | 1,260,884 | 17,102 |
| 4-connected components | 198 | 106 |
| Border-connected components / pixels | 1 / 1,260,096 | 4 / 16,770 |
| Largest component pixels / image area | 1,260,096 / 30.0430% | 16,762 / 0.399637% |
| Largest component bbox | [0,0,918,2048) | [0,0,144,217) |
| Interior components / pixels | 197 / 788 | 102 / 332 |
| Positive GT pixels on border / interior black | 0 / 0 | 0 / 3 |
| 8-connected components / interior pixels | 145 / 658 | 56 / 243 |
| <=5 border-connected pixels | 1,272,284 | 19,033 |
| <=10 border-connected pixels | 1,274,560 | 19,870 |

The exterior count here is the **entire connected component**, distinct from PR #25's outermost-row/column counts (3,391 and 354). Most of image 2's black area is one smooth left wedge reaching top and bottom; image 9's dominant black component is a straight-edged top-left triangle. Both lack normal scene texture. Enlarged transitions show a narrow near-black fringe; JPEG/YCbCr encoding and reprojection are plausible contributors, not a traced processing history. The max-channel median among nonblack pixels directly adjacent to exact-black pixels is 2 for 2, 1 for 9, consistent with why exact-zero connectivity fragments at the boundary. Higher values also occur; this is not a universal validity classifier.

Source/distribution/annotation-overlays and enlarged boundary/interior contexts were actually viewed. Image 2's sampled interior component sits near the wedge fringe, so “interior component” does not imply a real isolated ground object. For image 9 the largest sampled interior dark region is in a tree-shadow/building/road context. Three exact-black pixels are labelled canopy at (row,col)=(845,1947),(1840,1742),(1841,1742); enlarged canopy context was viewed. A black-pixel-only mask would remove annotated positives as well as suspicious fill. GT membership is not itself a validity guarantee, but demonstrates that black and annotation background are not equivalent. Image 2 remains uniformly background; no positive-mask alignment can be tested there. Image 9's main crown overlays broadly align, without establishing complete ecological GT or accuracy.

### Is enough usable context left?

Exact nonblack fractions are ~69.94% (2) and ~99.59% (9). These are **not certified valid fractions**. As a geometric upper-bound-style diagnostic, compute the largest axis-aligned square avoiding only border-connected <=5 regions, retaining all other pixels provisionally. No crop is written or accepted:

- 2: 1,386px square at [662,0,2048,1386), ~123.19m per side, about 246px at 0.50m. Under this hypothesis it does not support the existing 256px Train context without another policy change. This does not prove a valid crop exists.
- 9: 1,950px square at [96,95,2046,2045), ~146.07m, about 292px at 0.50m. Geometrically sufficient for a 224px Val crop, but validity and the crop boundary are not certified.

Near-black sensitivity can include legitimate shadows; the diagnostic is not an exact bound on true usable area. The larger nonblack rectangle can contain unknown invalid pixels. Certification of a fully valid region remains unresolved for both.

### TIFF and source-STAC findings

The **cached OAM-TCD tile originals**, not the upstream full OAM mosaics, were inspected byte-for-byte after SHA verification against PR #25. Both are 2048x2048, 3 uint8 RGB bands, EPSG:3395, north-up ~0.10m projected intervals. Affine/bounds remain those in the prior result. Compression tag 259=7 (JPEG), photometric tag 262=6 (YCbCr), tiled 512 square, AREA_OR_POINT=Area. No alpha/ExtraSamples, GDAL NoData tag 42113 or GDAL metadata tag 42112; no TIFF description/software tags. Pillow exposes one IFD frame; GDAL exposes no internal validity mask, no .msk sidecar and all band mask flags are all_valid. Dataset and band masks contain only 255. This describes what the decoder exposes, not a forensic guarantee about an upstream file.

[Rasterio's mask documentation](https://rasterio.readthedocs.io/en/stable/topics/masks.html#dataset-masks) explains that in the absence of explicit mask/alpha/nodata it returns an all-255 dataset mask. Thus this flag is a fallback, not independent evidence of complete image coverage. The same [official discussion](https://rasterio.readthedocs.io/en/stable/topics/masks.html#reading-dataset-masks) explains that valid 8-bit dark data may become zero. Neither marking every zero invalid nor accepting every 255 mask pixel resolves this case.

The hash-pinned external STAC snapshots contain bbox, MultiPolygon geometry, projection/file/alternate asset metadata, and visual/metadata/thumbnail assets. No explicit validity asset or raster-band nodata declaration was found. Their original asset CRSs are EPSG:32719 for 2 and EPSG:32617 for 9, rather than the distributed tile's EPSG:3395. Source proj:geometry is a projected rectangle; transforming its corners to WGS84 agrees with the Item geometry within 4.63e-7 and 4.60e-7 degrees respectively, consistent with rounded nominal raster extent. Full source mosaics/headers, actual alpha/NoData and optional historical .msk/.vrt files were **not** retrieved or inspected.

Rasterizing Item geometry at the tile's pixel centers gives:

- 2: 52,226 centers outside nominal Item geometry (all black), but **1,208,658 black pixels also lie inside it**.
- 9: every tile center lies inside Item geometry, including **all 17,102 black pixels**.

Consequently neither the catalog bbox nor this nominal geometry identifies the black fill. The [STAC 1.1 Item specification](https://github.com/radiantearth/stac-spec/blob/v1.1.0/item-spec/item-spec.md#geometry) describes an asset footprint; it does not certify the accuracy of these particular provider-supplied geometries as pixel-level valid masks. The measurements, rather than an assumption about the field name, determine our limitation here.

### Official material reviewed and causal confidence

Reviewed on 2026-11-05; scope is the following documents/code, not an exhaustive claim about all upstream material:

- [Fixed TCD dataset documentation, Data format](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/docs/dataset.md#data-format): tiles are sampled with a valid-pixel threshold. It does not state that every released tile is entirely valid or specify a recoverable zero/coverage mask rule there. This is compatible with some incomplete-coverage tiles surviving selection, but does not identify the valid mask of 2/9.
- [Fixed official GeoTIFF exporter](https://github.com/Restor-Foundation/tcd/blob/05dab754a3fd15b83d6b7606de7310155f4e2df2/tools/generate_dataset.py#L14): save_geotiff writes pixels/geometry without explicit nodata/mask handling. **It is not established that this exporter created the cached HF TIFF bytes.** It cannot prove which stage lost or introduced coverage information.
- [OAM data layout](https://docs.openaerialmap.org/ecosystem/imagery-data/): publishes COGs and metadata; mentions some older uploads also have .msk/.vrt files. [Uploader documentation](https://docs.openaerialmap.org/uploader/uploader-form/) describes contributed TIFFs and metadata. These pages do not provide a universal black-border/NoData validity guarantee. Optional sidecars are a future evidence lead, not a mask we have inspected.

| Region | User's causal category | Confidence and limit |
| --- | --- | --- |
| 2 dominant exterior wedge | **2: likely coverage-outside/fill, not confirmed** | High qualitative likelihood from shape, constant color and sharp terrain boundary; exact stage/source coverage is unknown |
| 9 dominant exterior triangle | **2: likely coverage-outside/fill, not confirmed** | High qualitative likelihood from geometric boundary against textured water; Item geometry does not resolve it |
| 2 small internal fragments | **4: unresolved**, often compatible with the same fringe | Exact-zero islands merge under near-black/8-connectivity; their status cannot be inferred from connectivity alone |
| 9 internal dark pixels | **3: real dark surface/shadow plausible** | Enlarged shadow/canopy context and three positive GT pixels oppose blanket zero masking; not every small component individually reviewed |

**Category 1 (confirmed outside-coverage fill) was not established.** No probability percentage is invented. We cannot distinguish upstream orthomosaic padding from later reprojection/resampling/RGB/JPEG serialization or a combination. The encoding fact does not prove the JPEG stage introduced the geometric wedge.

### A/B/C comparison and decision

| Option | Quality | Implementation / existing-code effect | Nationwide expansion implication |
| --- | --- | --- | --- |
| **A: exclude entire affected images; use other candidates** | Avoids invented labels/validity, loses useful pixels and may bias coverage if widely used | Low: maintain HOLD/exclusion audit and replan explicit candidates; no preprocessing/loader/loss change | Practical initial path while suitable alternatives exist; log exclusions and revisit if a larger audit shows material coverage loss |
| B: crop a proven valid region | Retains known-valid context only if evidence establishes it; changes extent/context; 2's hypothetical crop is too small for current 256px Train context | Moderate: review the coverage evidence and deterministic crop, matching RGB/GT/affine/bounds/GSD/manifests; reverify size/selection | Source-specific exceptions can be costly; useful for scarce/valuable sources, not justified just by these two cases |
| C: validity-aware labels/loss | Can retain more data with a reliable source mask; color-only validity can discard real shadows/canopy | High: source validity, valid-weighted resampling/partial pixels, loader, loss denominator, metrics and provenance/tests need a separately reviewed contract | Potentially reusable once authoritative masks/policies exist, but present metadata is insufficient and no incidence/benefit has been measured |

For C, the installed official Satlas 0.3.1 Head uses unreduced CE followed by mean over **all** output pixels; [official head source](https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/heads.py) was checked alongside the installed source. [PyTorch 2.7 CE](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.cross_entropy.html) defaults ignore_index=-100, but an ignored zero loss still participates in the Head's subsequent denominator. Existing test `test_official_ignore_default_is_zero_but_mean_diluted` covers this. Merely writing -100 is not a valid-area mean; all-invalid rejection, denominator/evaluation masking and invalid-context influence need review. Existing loader also rejects non-0/1 targets and valid_fraction !=1. Do not change the official architecture or silently override this loss contract here.

**No immediate validity-processing implementation is necessary to expand the pilot. Prefer A.** Propose Train 3 and Val 576 as the next unacquired approved-list entries, subject to a fresh footer/cost/rights/split plan. This is a proposal, not acquisition or training approval; neither was fetched here. Earlier Train 3's storage HOLD concerned the three-candidate budget, not observed image quality. Do not auto-increase budgets or fetch replacements in this investigation. Revisit B/C only if alternatives are insufficient, valuable sources require recovery, or a wider bounded audit demonstrates a material need and valid masks can be evidenced.

The current “one exact-black edge pixel => HOLD” rule should remain a **temporary review trigger**, not a permanent automatic exclusion or black-to-NoData rule. Rasterio's dark-zero warning, the observed interior positives and lossy fringe show why color alone is insufficient. A later triage proposal should prioritize explicit mask/alpha/nodata evidence, then inspect connected area/shape, near-black sensitivity and actual context. No universal pixel/percentage cutoff can be selected from two cases. Leave the production rule unchanged for this PR; record a manual, per-source decision when advancing candidates. HOLD and final exclusion are different decisions.

### Reproduction, tests and unresolved items

```powershell
# Existing pilot cache only; fresh external diagnostics directory required.
.\.venv\Scripts\python.exe scripts/m1_5b2_border_audit.py --cache '<existing-M1-5B-cache>' --output '<fresh-external-border-diagnostics>'
.\.venv\Scripts\python.exe -m pytest -q --disable-warnings
```

The diagnostic CLI is completely offline and checks raw_manifest plus every original/annotation/segment/STAC/preview hash against the prior public record. A raw/distribution/GT triptych and edge/interior details are created externally; none are masks for training. Geometry rasterization uses pixel-center membership, all_touched=False. The square calculation assesses only geometric capacity under a stated hypothesis and writes no crop. No dependencies were installed. Results are reproducible from the cache without Codex or ChatGPT.

Regression result: **106 passed**, including four new diagnostic tests, with 75 existing/reused Affine warnings. Tests independently compare run-union components with flood-fill on synthetic arrays (4/8 connectivity), distinguish exterior/interior and annotated dark pixels, check square capacity/threshold sensitivity, and validate empty/bad inputs. No real model training or GPU experiment was performed.

Unresolved: true source valid-data mask/footprint, exact processing stage of fill/coverage loss, source-side alpha/nodata/sidecars, per-component ecological interpretation, certified crop suitability, B/C semantics and corpus-wide incidence. Evidence that would establish category 1 or justify B/C would include an authoritative original coverage mask or documented processing trace spatially matched to the tile; a bbox or an RGB-zero threshold alone is insufficient. These unknowns do not block the reviewed A-first replacement plan, but they do block using 2/9 as all-valid training pairs.
