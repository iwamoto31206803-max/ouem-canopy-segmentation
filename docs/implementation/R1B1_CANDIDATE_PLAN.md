# R1b-1 — Bounded Zenodo candidate plan

Scope: planning only, on main after PR #42. No image/annotation download,
source raster decoding, pair generation, model execution or training acceptance.
The existing G0–G3-A data/code and R1a contract are unchanged.

## Frozen cohort and selection

The [machine-readable plan](R1B1_CANDIDATE_PLAN.json) inventories the locally
available Zenodo `11617167-v1.0.0` JSONL catalog: **4,608 images / 541 oam_id
groups; 4,169 official Train records**. Both JSONL SHA-256 values are checked
against the accepted M1-5D-2 receipts before selection. Official Test/Holdout
metadata is used only for separation checks; its images/masks are never opened.

The pool is **150 provisional images: Train 120 / Val 30**; it is not approval
to acquire or process 150 images. The first batch is **25: Train 20 / Val 5**:

| Split | Initial image IDs in selection order |
| --- | --- |
| Train | 1638, 1054, 2669, 2493, 689, 113, 878, 4198, 500, 351, 598, 169, 615, 2299, 910, 4526, 3100, 818, 717, 3156 |
| Val | 1241, 22, 5014, 4328, 5003 |

All **25 remain HOLD**. Two (500, 717) have cached STAC declarations passing
the existing license/ID/producer/licensor/original-object-key checks; 23 have
no cached STAC proof. Even the two are not R1 rights/QA accepted. The entire
150-image pool contains five cached declarations, with all other evidence gaps
explicit. None of the archived Zenodo10 is counted as R1 accepted.

Deterministic policy, seed **0**:

1. Preserve official Train folds 1–4 / Val fold 0. Require explicit row
   `CC-BY 4.0`, metadata identity, location and dimensions. Never derive split
   from the on-disk directory named `holdout`.
2. Exclude archived Zenodo10 and their source groups/original metadata object
   keys. Also exclude known historical HOLD sources; the plan records 17 held
   IDs (including 1, 2, 9, 576, 1401, 4235), without releasing any old HOLD.
3. Detect repeated oam_id and canonical metadata-object paths (bucket-independent
   path SHA-256 is published; private URLs are not). Choose one representative
   per original source using SHA-256 of `seed:oam_id:image_id`, then image ID.
   Same-source role/fold conflicts are rejected. Source aliases not represented
   by this metadata key remain an external-review obligation.
4. Reuse `m1_5b_pilot.separation()` against **all** opposite-role source groups,
   including official Test metadata. Aggregate known tile envelopes per source;
   reject overlap/identity conflict and conservatively HOLD geodesic proximity
   lower bounds below 1km. This is metadata screening, not proof of separation
   between complete original STAC footprints or unlisted neighboring scenes.
5. Fill four Train slots then one Val slot, preferring the least-used 10-degree
   cell, then least-used biome, then seeded SHA rank. The first 25 form a stable
   prefix of the larger plan. No candidate is substituted automatically. Every
   catalog image has a reason under `reasons_by_image`; duplicate rows and pool
   reserves are distinct from quality rejections.

The initial cohort covers **25 geographic cells / 15 metadata biome values**.
Canopy fractions and scene/quality conditions are **NOT_MEASURED/NOT_VERIFIED**;
we do not claim diversity in those unobserved attributes. R1b-2 records their
distribution after screening this predeclared cohort. Do not select retrospectively
by Val scores, predictions, apparent urban character or canopy abundance; preserve
uniform masks. Any revised stratification requires a new reviewed plan.

`plan.completeness` separately checks actual pool and initial-batch totals and
Train/Val counts. For N slots, expected Val = floor(N/5), Train = N-floor(N/5),
following the declared four-Train/one-Val order (21 slots = 17/4). Missing candidates
or wrong split counts produce completeness HOLD, `gate="HOLD_INCOMPLETE_COHORT"`,
a blocker and CLI exit **2** after writing a diagnostic report. A complete initial
batch cannot hide a short pool, or vice versa. Estimates on incomplete plans cover
only the available subset and never authorize the requested complete batch.

Candidate fields explicitly distinguish `representative_metadata_bounds_wgs84`
and `representative_metadata_center` (selected tile; diversity uses its center)
from `group_metadata_bounds_wgs84` and `group_metadata_center` (union of known
TCD tiles sharing oam_id). `group_metadata_member_count` records aggregation size.
Separation uses the group union, not just the representative. Neither envelope is
automatically a complete original/STAC footprint. The ambiguous old
`metadata_bounds_wgs84` / `metadata_center` names are removed.

Plan ID: `ouem-r1b1-fbb4ed018dc3cfded676d6ba09229ef7ccfb80fa7bef9b9b67a1df84864fcc5a`.
It is `ouem-r1b1-` plus SHA-256 of the complete report before adding `plan_id`,
using the existing G0 `encoded()` serializer (sorted keys, ASCII escaped,
indent=2, UTF-8, final newline). This is a **plan** ID, not an R1 dataset ID.
Two fresh runs produced byte-identical reports, file SHA-256
`47263941638588f2a94e462ff04d247cd94df2426c92e42b86ac1a43a20e7ffe`.
This byte SHA describes the generated LF reports; Git text checkout may change
line endings. The canonicalized plan ID is the cross-platform content identity,
not a promise that a checked-out text file retains that byte SHA.

This supersedes plan `ouem-r1b1-9e721bd3d064910606bb8a43d89365bb70daeb82027553d766d6065f778a51cb`
(original PR commit a9e7387, retained in Git history). Completeness receipts,
explicit tile/group fields and the resource execution gate changed the ID. All
150 selected IDs/order, including the initial 25, are unchanged. The report's
`supersedes_plan_id` and `revision_reason` record this lineage; the optional CLI
argument adds the predecessor only when explicitly supplied.

## Resource model and proposed ceilings

Local read-only inputs already exist for all 25. Stat-only measurements:

| Evidence | Measured bytes |
| --- | ---: |
| Initial 25 RGB TIFFs, total | 17,859,995 |
| Initial 25 annotation PNGs, total | 883,541 |
| Existing 10 processed pairs (40 TIFFs), total / largest pair | 3,588,592 / 564,240 |
| Existing 18 QA previews, total / largest preview | 6,809,925 / 583,384 |
| Existing 2 run manifests, total / largest | 47,517 / 33,043 |
| Existing 19 STAC files, total / largest | 65,418 / 8,042 |

These are filesystem byte sizes, not new content or rights validation. Prior
pair-manifest hashes and the historical QA ledger hash bind the evidence sources.
The 19 STAC file count may contain repeated sources across caches; it is not a
count of newly accepted rights records.

For comparison, M1-5B-3 used **154,311,469 HTTP payload bytes** for two targets,
including 154,266,320 compressed column bytes and a largest row group of
78,838,830 bytes. Saved bytes at observation were 4,215,279; peak disk usage was
NOT_MEASURED. These historical Parquet costs are **not extrapolated** to Zenodo.

Proposed first-batch envelope, **subject to review; not an assertion of fit**:

* Image/annotation network: **0 bytes** using the existing immutable local cache.
  HTTP Range/Parquet cache: **0 bytes**. Missing cache/member => STOP, no fallback.
* Additional metadata network: at most **2 MiB (2,097,152 payload bytes)**,
  max 25 STAC requests, at most 65,536 bytes per response. Cache reuse is preferred;
  failures/retries consume the same cumulative budget. No automatic archive fetch.
* Incremental disk ceiling proposal: **94 MiB (98,566,144 bytes)**, separate from
  the already-existing canonical cache. The modeled peak is **48,948,126 bytes**:
  two deterministic output runs 28,212,000; previews 14,584,600; report/manifest
  allowance 2,692,444; STAC snapshots 1,638,400; temporary files 1,820,682.
  Apply 2x headroom, round upward to whole MiB. Reports reserve twice the encoded
  selection-plan bytes plus two historical maximum manifest sizes per image.
  Temporary allowance is the largest selected raw TIFF+PNG pair plus one observed
  maximum processed pair. Do not duplicate all raw sources into a new cache.

**Peak usage has not been measured for this cohort.** Compression, output dimensions,
error logs, additional detailed previews and temporary-write behavior can differ
from ten samples. A smaller approved ceiling is allowed; exceeding either ceiling
must STOP/HOLD, never expand it automatically. R1b-2 must meter writes/reservations,
confirm actual free space and stop before committing a write that cannot fit.
The planner only proposes caps; it does not execute/enforce future acquisition.

For a future PC without this cache, archive sizes are separately recorded:
images 2,977,187,901; masks 140,227,720; metadata 354,014; train COCO archive
99,442,722 bytes. Full-cache bootstrap/expanded peak capacity and a new transfer
approval are **not estimated or authorized by the 2MiB/94MiB pilot envelope**.
Gzip tar archives are not a cheap per-image HTTP Range replacement. Stop and
make a separate bootstrap plan; do not revive historical HF access silently.

This planner performed **0 dataset HTTP requests / 0 payload bytes**, and opened
**0 source rasters**. Git fetch/push transport is separate and NOT_MEASURED.

## QA checklist and R1a evidence mapping

Every first-batch image receives manual review, including automated HOLD cases
where the original can safely be inspected. Automated failure is not converted
into PASS. Later audit density is a new decision after reviewing this batch's
misses/yield; do not pre-assume that random sampling alone suffices.

| Item | Automatic evidence | Human review / R1a connection |
| --- | --- | --- |
| Rights/source | Explicit row license; hashed STAC ID, CC BY, object-key and named producer/licensor checks | Confirm scope, original source rights and actual attribution; populate `rights` + source-bound `r1_review`; retain contacts only externally |
| Bytes/generation | Archive/member authentication, SHA-256 of original TIFF/PNG, exact metadata filename ID | Resolve contradictory archive/source history; populate `source`, never copy HF hashes |
| Labels | RGB uint8 PNG, dimensions, COCO categories 1/2; official nonzero union; fractions and inclusive 0.5 | Assess label plausibility/completeness; `labels`, `category_review`; per-segment crosswalk remains NOT_VERIFIED if absent |
| Validity | TIFF masks/nodata/alpha, black connected components, anomalous values | View RGB, mask, overlay and suspicious boundaries for every image; `validity_review`; ambiguous black wedge HOLD, no black-to-NoData inference |
| Alignment/scale | Actual embedded TIFF CRS/affine/bounds, original/PNG alignment, local-ground ~0.50m grid, saved/readback product SHA | Inspect before/after overlays; no fixed factor-5 shortcut or unapproved geometric repair |
| Internal separation | oam_id/object identity/fold and metadata envelope/proximity checks | Establish actual canonical original-source identity/extent; `independence_review` |
| External separation | Compare complete source catalog metadata, official Test/Holdout and neighboring/alias sources; log covered universe and unresolved entries | Issue `external_independence_review`: scope `official_holdout_and_unlisted_neighbor_sources`, `catalog_sha256`, evidence ref/SHA and matching original TIFF/PNG hashes; missing catalog coverage HOLD |
| Final R1 gate | Four products and all-valid checks; two fresh output runs SHA-equal; immutable R1 manifest | `r1_review` is separate from historical smoke acceptance and final commercial approval; `r1_dataset.verify()` + `preflight()` must pass |

Each review receipt uses R1a's `status`, `evidence_ref`, `evidence_sha256` and
`source_hashes` (`image_sha256`, `annotation_sha256`). External review additionally
pins the inspected metadata universe/catalog SHA and states its coverage/aliases,
method and unresolved gaps. An offline declaration-completeness check is not a
fresh independent legal or external spatial investigation. Archive-level CC BY
does not fill unknown per-record/source rights. Preserve all-background/all-canopy
teachers and record their rates rather than forcing mixed labels.

## R1b-2 execution order and restart protection

**Not execution-ready yet:** the report explicitly records blockers. Candidate
planning succeeded; it does not authorize bulk image processing or declare QA PASS.

1. Review this plan ID/cohort, proposed ceilings, evidence gaps and local free space.
   Freeze the campaign ID, input hashes, allowed 25 IDs and scope; no pool-wide run.
2. Implement/review a small persistent campaign ledger before any new HTTP/write:
   reserve bounded bytes **before** each operation, record actual payload including
   failures/partials, retained file bytes, reservations and hashed artifacts. Use
   exclusive operation records/lock and atomic publication. On ambiguous interruption,
   charge the unresolved reservation and STOP for reconciliation; do not replay a
   request, delete a stale lock, reset counters or switch to a fresh allowance.
   Resume must match plan/input hashes and reconcile completed/partial files.
   Only owned, hash-checked temporary paths may be cleaned after review.
   The machine-readable `resource_execution_gate` remains HOLD with
   `actual_batch_peak_measured=false`. R1b-2 must measure retained/temporary bytes,
   persist cumulative transfer/storage usage and STOP **before** the next HTTP
   request/write/reservation exceeds either approved limit. The 94MiB proposal
   is not an observed peak or automatic execution permission.
3. Collect only missing, bounded STAC evidence (or reuse verified snapshots).
   Publish sanitized evidence references, not contacts. Resolve rights/identity
   conflicts before creating training pairs. Network failure remains HOLD.
4. Screen and manually inspect the **fixed** 25 IDs, using existing low-level
   label/geometry/border/overlay functions. Important: the old `--screen` CLI enforces
   its historical fixed 18-image candidate list and **cannot run this cohort**.
   R1b-2 needs a small plan-bound screening wrapper/extension; do not bypass its
   checks or modify archived screening receipts. It must share the campaign budget.
5. For source-bound approved evidence only, reuse the existing Zenodo preparation
   path with explicit IDs and fresh output roots. Its historical `PASS_CANDIDATE`
   is not R1 acceptance. Run a second distinct output root and compare all SHA.
6. Build a **new R1 manifest** from the actual products/rights/review receipts;
   seal its dataset ID and file SHA, run `r1_dataset.verify()` and `preflight()`.
   Report accepted/HOLD/FAIL by image. The existing Zenodo10 remains untouched and
   separately HOLD for R1 until independently requalified.
7. Review observed yield, canopy/scene distributions, missed defects and measured
   peak usage before authorizing any next bounded cohort. No automatic replacements
   or budget increases. A new plan cannot reset the campaign's used allowance.

## Offline Windows reproduction

Set `$Repo`, `$Cache`, `$Audits` to local directories; use the existing venv.
No Codex, API key or network connection is needed by this script.

```powershell
Set-Location $Repo
.\.venv\Scripts\python.exe scripts/r1b_candidate_plan.py `
  --zenodo-root "$Cache/oam-tcd-zenodo-v1" `
  --ledger docs/implementation/M1_5B_PILOT_RESULT.json `
  --g0-manifest docs/implementation/M1_7_G0_MANIFEST.json `
  --evidence-dir "$Cache/m1-5a-metadata-03" `
  --evidence-dir "$Audits/m1-5d3-rights-01" `
  --evidence-dir "$Audits/m1-5d3-rights-reserve-01" `
  --pairs-dir "$Audits/m1-5d2-pairs-01" `
  --pairs-dir "$Audits/m1-5d3-pairs-01" `
  --preview-dir "$Audits/m1-5d3-screen-01" `
  --preview-dir "$Audits/m1-5d3-reserve-01" `
  --seed 0 --pool-size 150 --batch-size 25 `
  --supersedes-plan-id ouem-r1b1-9e721bd3d064910606bb8a43d89365bb70daeb82027553d766d6065f778a51cb `
  --output $NewPlanPath
```

Output requires a fresh file outside protected data/evidence directories; existing
reports are not overwritten. Missing measurements => no disk estimate/HOLD, not
fabricated numbers. Exit 0 means a **count/split-complete plan was written**, not
candidates accepted or the R1b-2 gate released. A shortfall writes HOLD and exits
2. Invalid input fails without claiming completeness.
Different evidence/measurement inputs produce a new plan ID.
No old scripts, manifests, cache files, checkpoints or dataset acceptances change.

Tests use synthetic metadata and byte-size fixtures, not real image QA. They check
repeatability/prefix/seed behavior, fold/holdout/legacy/source separation, missing
rights, metadata generation/hash pins, budget arithmetic, and output preservation.
Actual pixel/category/validity/visual/rights/source-byte audits remain R1b-2 work.

Measured on Windows/Python 3.11.9 with the existing venv:
`python -m pytest -q` → **289 passed, 1 skipped, 676 warnings (67.88s)**;
new planner tests → **22 passed (0.95s)**. The skip is the existing Windows
symlink-creation limitation; warning categories are existing rasterio/Affine
pending deprecations and the existing duplicate-ZIP negative case.
Two complete local planning runs matched byte-for-byte, and the G0 manifest
retains its fixed SHA `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca`.
Review tests additionally cover Train/Val shortage, a short pool with a complete
first batch, nonmultiple-of-five slot quotas, CLI HOLD/nonzero exit, and the
actual aggregate envelope passed into separation. Candidate IDs/order were
compared to a9e7387 and are unchanged; the regenerated report is byte-identical
across two new runs. Both count/split checks pass for the real 150/25 cohort.
