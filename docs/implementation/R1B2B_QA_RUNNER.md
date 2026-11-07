# R1b-2b — Approval-bound QA Runner and I/O adapters

Implementation/offline tests only. **No production approval was issued**, real
campaign initialized, STAC fetched, real 25-image batch decoded, training product
generated, or model/GPU used. Existing G0–G3-A, R1a, R1b-1/R1b-2a code, manifests,
data and Zenodo10 R1 HOLD are unchanged. This Runner never emits R1 acceptance PASS.

## Entry points and default deny

`scripts/r1b_qa_runner.py` contains one small plan-bound Runner and its approval,
STAC and managed-save adapters. It uses the merged Campaign contract rather than
adding a second ledger. No new dependencies or Notebook are introduced.

The production CLI exposes only `check-approval` and `run`. Both require an
external approval file plus its independently reviewed expected SHA. `run` also
requires `--execute`; fetching missing STAC additionally requires the explicit
`--fetch-missing-stac` flag and `stac_network` scope. Missing approval, wrong mode,
binding, environment, dirty production checkout or missing scope stops before
source-image/HTTP access. `check-approval` reads control data only and refuses a
campaign in STOP; it does not create an approval or run any QA.

The approval is a trusted operator/reviewer artifact, **not a cryptographic proof
of human authority**. Obtain its expected SHA from the independent approval
process; computing the hash of an arbitrary self-authored `approved:true` record
does not constitute review. There is no approval-generation command. Unit tests
create explicitly `offline_test` receipts usable only through explicit test-mode
API plus mock transport; the production CLI rejects them and has no test-mode flag.
Test mode additionally requires a SHA-bound `SYNTHETIC_QA_FIXTURE.txt` marker with
the exact synthetic-only contents in its fake cache, and rejects live stdlib
HTTP connection classes. Never add this test marker to the canonical dataset.
The test API assumes a trusted mock callback, not a sandbox for arbitrary Python;
all tests independently block real socket connections.

### Approval contract: `ouem-r1b-qa-approval-v1`

| Field | Required value/binding |
| --- | --- |
| `approved` | Boolean true only after independent execution approval; absent/false denies |
| `approval_id`, `reviewer_ref` | Recorded approval identity and reviewer reference; keep private receipts external |
| `mode` | `production_qa`; `offline_test` is reserved for synthetic harnesses |
| `campaign_id`, `contract_sha256` | Exact frozen campaign and contract bytes |
| `plan_id`, `plan_sha256`, `targets` | Exact campaign plan and full 25-ID Train/Val map |
| `limits` | Exact frozen network/request/response/storage limits; no implicit changes |
| `cache_root`, `output_root` | Exact fixed cache and resolved campaign destination |
| `environment` | Exact `environment()` fingerprint described below |
| `scopes` | Explicit subset of `qa_local`, `qa_preview`, `qa_report`, `stac_network`, `geotiff_adapter` |
| `artifact_limits` | Positive byte caps for preview/report/stac/geotiff, each <=32MiB; STAC <=campaign response limit |
| `max_input_pixels` | Positive, <=2048², checked against image headers before array decode |
| `max_buffer_bytes` | Positive, <=32MiB, per managed encoded buffer / TIFF allocation-estimate guard |
| `coco_sha256` | Pin of official Train COCO JSON before streaming category inspection |
| `local_stac` | Optional target-ID map of absolute snapshot path and SHA; unknown IDs/invalid pins deny |

`environment()` records Python/platform, numpy/Pillow/rasterio/pyproj versions,
Git commit/dirty status and this Runner file's byte SHA. Production requires a
clean checkout; approval must be prepared for the reviewed merged commit and
actual Windows environment/line endings. Contract/targets/limits/environment
comparisons use serialized JSON equality, avoiding bool/int coercion. Approval
file or in-memory record changes are rejected before subsequent managed actions.
This is session-bound checking, not a monitor of arbitrary concurrent code changes.

A **non-executable** review template starts with `approved:false`, no real reviewer
or approved hashes. Populate/bind the fields above through a separately reviewed
execution process; this PR does not change it to true. Collect environment details
read-only if needed:

```powershell
.\.venv\Scripts\python.exe -c "import sys,json; sys.path.insert(0,'scripts'); from r1b_qa_runner import environment; print(json.dumps(environment(),indent=2))"
```

## Plan-bound local QA

Runner creation rechecks plan SHA/ID, both Zenodo metadata SHA pins and the fixed
cohort, then uses `local_index()` and a SHA-pinned streamed Train COCO summary.
Source paths come **only** from `contract.source_files`; metadata oam_id/image_id,
fold/split and exact TIFF/PNG filenames must agree. No free cache scanning, image
substitution, Holdout pixels or GT54 are permitted. Metadata for official Test is
used only for separation; directory name `holdout` is never used to assign roles.

The initial default processing order is the plan's first-25 order, not JSON map
key order. An explicitly requested subset is checked for duplicate/unplanned IDs;
the summary lists IDs not processed in that invocation, not QA-complete. Earlier
per-image reports remain in Campaign history; the summary is not an automatic
multi-run acceptance merger.
Deterministic operation names reject repeat image work before source decode.
There is no automatic replay/skip/replacement scheme for a partially completed
cohort: use Campaign inspect/reconcile and obtain a reviewed follow-up strategy.

Low-level reuse (the historical 18-image CLI is unmodified):

* `semantic_binary()` / `coco_annotation_summary()` enforce RGB uint8/dimensions,
  categories within 1/2 and the official `any(panoptic_rgb != 0)` union. PNG IHDR
  also requires actual 8-bit RGB without declared transparency, so a decoder's
  normalization cannot silently accept a different source bit depth.
* Source bytes are streamed through the existing SHA helper. TIFF requires RGB
  uint8, EPSG:3395, finite north-up affine, expected shape/bounds and rasterio/Pillow
  pixel agreement. No resampling, orientation repair or source writes occur.
* `component_report()` measures border-connected versus internal exact-black
  regions. Existing >=1% connected border-black heuristic gives automatic HOLD;
  small dark pixels alone do not fail. No automatic black-to-NoData conversion.
* Per-band masks/nodata/alpha are checked. Metadata masks being all valid is not
  proof of complete geographic image coverage; **no ALL_VALID visual receipt or
  valid_fraction image is fabricated**.
* `intervals()` / `output_grid()` report source ground GSD and proposed ~0.50m
  output dimensions. This QA path does not generate reduced training products.
* Existing `separation()` checks aggregate metadata group envelopes against all
  opposite-role groups (including Test metadata). Representative TIFF geometry
  and candidate group envelope are recorded separately. Source identity duplicates
  among the 25 also cause HOLD. Complete original/STAC footprints, aliases and
  unlisted neighbors are still external-review obligations.

RGB, union mask and overlay are assembled into a deterministic three-panel JPEG
(quality92, max1800x600), matching the older pilot's JPEG approach. It is a lossy
review diagnostic, not a changed teacher or numerical validation artifact. Uniform
0/1 masks stay unchanged; their source fractions are reported. Detailed manual
inspection of original images may still be needed; no visual PASS is inferred.

## STAC adapter

`runner.stac.fetch(image_id, split)` accepts only the exact HTTPS official item URL
for the target's oam_id. Override URLs, query strings, alternate IDs, asset/image/
Parquet URLs, HTTP and credentials are not followed. Before constructing/sending
the connection, reserve one request, the full frozen response allowance and the
approved output cap through Campaign. The endpoint has no automatic retries,
redirect following, proxies or asset traversal: stdlib `HTTPSConnection`, one GET,
explicit identity encoding, connection close. Mock injection is test-mode-only.

The response class replaces HTTP's buffered makefile with an unbuffered reader
**before** reading headers; no `read()` without a bound or preload of the whole
body is used. Require status200, one numeric Content-Length strictly below the
response allowance and within the output cap, one JSON content type, no transfer
encoding, and at most one identity content encoding. Duplicate/conflicting headers,
gzip/chunked, missing length, redirects, truncated content or EOF not proven before
the bound cause STOP. All failures within Campaign.receive retain full reservation
and pending state; connection closure does not refund unknown consumption. A
misbehaving reader's over-return reaches Campaign's fatal accounting check.

This bounds **application response-body bytes**, consistent with R1b-2a. Header,
TLS-record/socket buffering and total wire bytes are not included or measured;
unbuffered HTTP does not claim to disable TLS/kernel buffers. Actual HTTPS/server
behavior is NOT_VERIFIED here. Python http.client internals and the environment
fingerprint must be revalidated on runtime upgrades.

Raw responses are saved only via Campaign, then strict JSON/Feature/version/ID,
closed Polygon/MultiPolygon rings, WGS84 bbox, source asset links and original
metadata object identity are checked. Existing `rights()` requires explicit row
CC BY4.0, matching STAC license/ID, named producer/licensor and original metadata
key. No asset is downloaded. A saved STAC's I/O SUCCESS is not rights acceptance:
invalid/incomplete declarations remain HOLD in QA. Cached STAC must be individually
pinned in approval, or have a completed Campaign STAC operation with matching SHA.
No arbitrary external evidence file can be substituted. Raw STAC/contact details
stay external; reports retain sanitized references/hashes, not provider contacts.

## Managed storage and memory

`Saved.encode/report/preview/geotiff` require the relevant approval scope and reserve
the approved maximum **before** encoder execution. They use Campaign write_bytes
and WRITTEN-receipt commit; no direct encoder writes to disk are allowed. A bounded
BytesIO prevents crossing the encoded-output cap. JSON is serialized incrementally;
Pillow writes into that buffer. The future GeoTIFF adapter uses MemoryFile, validates
arrays, checks a conservative `2*array.nbytes+65536` allocation estimate, measures
encoded size before copying bytes, then uses the same Campaign publication path.
It has only been tested with synthetic rasters, not used to generate real pairs.

On buffer/memory overflow, preserve the original reservation and STOP. A measured
HOLD diagnostic (observed/limit, peak NOT_MEASURED) is attempted as a **separately
budgeted** report respecting the approved report cap. If even it cannot fit, fail
without bypassing the quota; the original pending ledger remains. Storage/free-space
or partial-write failures also stop, without automatic retry or cap expansion.

Decoded arrays are header-pixel bounded, and reports record known array/encoded
byte sizes. `max_buffer_bytes` does not bound total native/Python RSS: copies,
connected-component arrays and decoder internals add memory. Real process/disk
peak and the ability to complete 25 images under94MiB remain NOT_MEASURED. An
abrupt kill may prevent a diagnostic; Campaign's pending reservations preserve STOP.

## QA report and R1a connection

Reports separate technical PASS/HOLD/FAIL, NOT_VERIFIED human/external items and
**always-HOLD R1 acceptance**. A clean automated raster check never promotes data.

| R1a field | Runner evidence and missing gate |
| --- | --- |
| `source` | Actual TIFF/PNG hashes, filename/metadata identity, generation, source-key hash; archive-member authentication still HOLD |
| `rights` | Hashed STAC declaration checks or HOLD; final source/image/annotation rights and attribution review absent |
| `category_review` | COCO category set and official binary rule; per-segment mapping and source-bound human review NOT_VERIFIED |
| `validity_review` | Mask/nodata/black components plus preview; actual visual all-valid/alignment review NOT_VERIFIED |
| `independence_review` | Grouped metadata/duplicate guards; full original-source identity/extent review NOT_VERIFIED |
| `external_independence_review` | Official Holdout/unlisted neighbor catalog coverage and review NOT_VERIFIED |
| `r1_review` | NOT_VERIFIED; cannot be issued by this Runner |

Known source hashes are attached to draft review evidence; `missing_evidence`
names each incomplete field. These are **not valid PASS receipts or a finalized
R1 manifest**. COCO annotation IDs are never equated with panoptic segment IDs.
No category crosswalk is invented and background0 is not treated as nodata.

Each preview/report has a fixed operation name, path, size/SHA and Campaign receipt.
The final managed `qa-summary.json` binds campaign/plan/approval, requested/results/
unprocessed IDs and a usage snapshot **before** the summary write. Campaign.inspect
gives later complete usage. Interrupted runs may have only prior per-image reports
and the ledger, not a final summary; STOP is never a completed cohort PASS.
CLI exits 1 for completed runs containing technical FAIL, otherwise2 because R1
acceptance remains HOLD. Approval check exits0 for binding validation only.

## Windows procedure — future approved execution only

Do not execute the production batch as part of this PR. Keep a reviewed Campaign
and permanent registry from R1b-2a, the fixed plan, cache and approval outside Git.
No approval is auto-created and no actual campaign was initialized here.

```powershell
$Plan = "docs/implementation/R1B1_CANDIDATE_PLAN.json"
$PlanSHA = (Get-FileHash $Plan -Algorithm SHA256).Hash.ToLowerInvariant()
$ApprovalSHA = "<expected SHA supplied by independent execution review>"
.\.venv\Scripts\python.exe scripts/r1b_qa_runner.py check-approval `
  --root $CampaignRoot --registry $PermanentRegistry `
  --plan $Plan --plan-sha256 $PlanSHA `
  --approval $ReviewedApproval --approval-sha256 $ApprovalSHA
# Only after explicit review/approval and a CLEAN_PAUSED campaign:
.\.venv\Scripts\python.exe scripts/r1b_qa_runner.py run --execute `
  --root $CampaignRoot --registry $PermanentRegistry `
  --plan $Plan --plan-sha256 $PlanSHA `
  --approval $ReviewedApproval --approval-sha256 $ApprovalSHA
# Add --fetch-missing-stac only when that scope/network budget is explicitly approved.
```

No source/old-manifest writes, hidden dependency installation, acquisition bootstrap
or automatic resume are performed. After any exception inspect the Campaign;
unresolved operations require the existing explicit reconciliation procedure.
Do not delete locks, reset the registry, change operation IDs to silently retry,
or auto-create a second campaign. Approval changes need a new independent receipt.

Remaining production HOLD: issuance/review of the real execution approval, real
HTTPS behavior, source/archive authentication and all25 source hashes, each image's
manual QA, rights/attribution and external-source independence, actual resource
peak/budget fit, and later separate R1 review/manifest/preflight. Models, inference,
training and automatic R1 PASS are out of scope.

## Verification record

Windows / Python3.11.9, existing venv, no installations:

* Full pytest: **375 passed, 1 skipped, 2023 warnings (155.48s)**.
* New Runner tests: **52 passed, 1347 warnings (63.40s)**.
* The skip is the existing Windows real-symlink-creation restriction. Warnings
  are rasterio/Affine pending deprecations (many tiny synthetic TIFF fixtures)
  plus the existing duplicate-ZIP negative case. Existing Campaign Windows
  locking/publication/crash tests and all other regression tests passed.
* Coverage includes approval binding/mode/marker/scope denial, fixed ID/split/URL,
  mock single GET/no retry/redirect, unbuffered HTTP parsing, encoding/header/EOF/
  partial receive, STAC source/license/geometry/JSON failures, pinned local evidence,
  uniform labels, dimensions/category/grid/bit-depth/dtype, black border, managed
  JPEG/JSON/MemoryFile TIFF save and SHA, cap/free-space/partial-write interruption,
  diagnostic-report cap, duplicate operation, subset summary and R1 HOLD.
* Tests block real socket connections. Synthetic approvals are not production
  approvals. The CLI happy-path test injects test mode only through pytest;
  production CLI has no bypass flag. No real STAC or image processing was tested.
* CLI help and `git diff --check` passed. The fixed G0 SHA remains
  `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca`;
  this checkout's unchanged R1b-1 plan SHA remains
  `98abe4b98cb629c3c596f70718db2854a2df675f6c7781792823f33f6115fcbc`.
