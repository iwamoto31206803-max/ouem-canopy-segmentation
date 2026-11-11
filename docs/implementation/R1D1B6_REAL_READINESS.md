# R1d-1B-6 fixed-four decision and execution readiness

The baseline is merged PR #61 at `9abf346c8bc709c6e1cc8b475ea4f1fb3079a3b2`.
This package implements the evidence-only preflight option in Issue #62.
G3/G4 remain PROPOSED and human authorization remains PENDING. All 150 Formal R1
states remain HOLD. The original synthetic conversion entrypoint is unchanged.

## Decisions for the present internal research use

| Gate | Technical recommendation | Required human decision |
|---|---|---|
| G3 | Yes, adopt the finite catalog review for Train 717/1054 and Val 1241/461 with the recorded residual risks | Accept the named catalog, aliases, geometry methods and unresolved gaps for these four |
| G4 | Yes, adopt the documented operational whole-tile coverage review for these four | Approve the operational whole-tile assertion and explicitly authorize its legacy `ALL_VALID` representation |

These recommendations reuse the source-bound archive, rights, independence,
coverage and independent-review receipts. Exact oam IDs, folds, original image
and annotation hashes, individual STAC hashes and providers are in
`R1D1B6_DECISION_PACKAGE.json`. The receipt references contain relative audit
paths and SHA-256 pins. Receipt existence and matching hashes support traceability.
Their substantive findings and the human application decision remain distinct.

G3 covers the fixed official Train 4,169/Test 439 metadata records, 541 groups,
17 known STAC bboxes, group-envelope proxies, exact object-path aliases, the
conservative 1-km guard and the existing Zenodo10 comparison. Existing evidence
reports no known cross-role leakage within that scope. Full source extents of
524 groups, undiscovered aliases and catalog-external overlap remain unresolved.
STAC bboxes and group envelopes carry their documented geometric meanings.

The R1a contract requires a source-bound external review with scope
`official_holdout_and_unlisted_neighbor_sources` and a catalog/index digest. It
records the R1b assertion and checks its completeness. The verifier leaves its
own external investigation status `NOT_VERIFIED`. The proposed finite review can
populate that assertion following explicit human interpretation of the bounded
scope. The current schema can carry the decision without amendment. Human
rejection of that interpretation leaves G3 on HOLD with a named missing fact.

G4 uses the existing full-frame RGB/teacher overview, boundary diagnostics and
bounded native-resolution inspection, with coordinates, scale and source/preview
hashes retained in the coverage receipt. The inspection found no major coverage
anomaly in its specified scope. `ALL_PIXELS_PROVEN_VALID` remains `NOT_VERIFIED`.
The actual adapter checks `review.coverage == "ALL_VALID"` and constructs an
all-one source validity array. Its later area aggregation therefore yields
all-one `valid_fraction`. An explicit user-approved operational whole-tile
assertion is needed for that representation. This preflight never converts
`REVIEW_PASS` into `ALL_VALID` and emits no R1a review object.

The teacher policy remains the upstream nonzero RGB panoptic union, with COCO
categories 1/2 and preserved all-zero/all-one masks. The per-segment category
crosswalk remains `NOT_VERIFIED`. Future review objects must retain that limit.

## Actual catalog bytes and review-object connection

`R1D1B6_CATALOG_INDEX.json` materializes the existing B4 proposal payload as
UTF-8, sorted keys, `ensure_ascii=False`, compact comma/colon separators, with
no BOM or final newline. Its 5,265 actual bytes hash to
`0bb9e9501792669d5365a25d11a6e354c410cc72b5cdf4cba73ac784ddfe37a0`.
Its historical method-code hashes identify the earlier investigation version.
Both new JSON artifacts have `-text` attributes to preserve pinned bytes across
Windows and Unix checkouts.

A later authorized source-bound review can reference this catalog and the
receipt hashes, exact image/annotation hashes, source generation, review scope,
intended use and human reference/date. The decision package can be hashed before
that separate authorization is created. The package contains no self hash.
Changing evidence or use requires a freshly reviewed binding. G1/G2 documentary
findings are reused, with the existing MD5/publication and future-use limitations.

## Read-only CLI

The CLI checks package bytes, fixed identity/order/split/fold/shape, filenames,
source pin syntax, category policy, unchanged limits and actual canonical catalog
bytes. An optional audit root additionally checks five pinned JSON receipts,
eight archive-member bindings, rights declarations and source identity bindings.
It opens only bounded JSON evidence files and rejects unsafe/reparse paths.
Embedded receipt paths, including original images, archives and previews, are
never followed. Evidence references have a 4-MiB read ceiling each.

```powershell
$PackageSHA = (Get-FileHash docs/implementation/R1D1B6_DECISION_PACKAGE.json -Algorithm SHA256).Hash.ToLower()
.\.venv\Scripts\python.exe -B scripts/r1d_real_readiness.py `
  --package docs/implementation/R1D1B6_DECISION_PACKAGE.json `
  --package-sha256 $PackageSHA `
  --catalog-index docs/implementation/R1D1B6_CATALOG_INDEX.json `
  --evidence-root 'C:\dev\ouem-data-audits'
```

Pin the expected package SHA from the reviewed PR when operating later. Computing
it locally in this example shows how to reproduce the measurement. Omitting the
audit root returns an empty `receipts_checked` map. A successful check always
returns `EVIDENCE_PACKAGE_CHECKED_EXECUTION_BLOCKED`, `execution_enabled=false`
and `source_bytes_checked=false`. Exit 0 describes this evidence check. The CLI
provides no execute option, output directory or image processing path.

The local evidence-only check succeeded for all five original receipt hashes.
No original TIFF/PNG bytes or archive bytes were read in this task. The expected
real output shapes 390²/391²/394²/337² are pinned proposals for later grid checks.

## Offline boundary assessment

Read-only host inspection on 2026-11-11 found Windows Professional 26H2,
build 26300.9818. `WindowsSandbox.exe` was absent at the checked system path.
The optional-feature query required elevation, so the installed feature state
remains unverified. Firmware virtualization reported true and VM monitor mode
extensions false. These observations leave guest prerequisites unresolved.
The current Python 3.11.9 venv uses host-absolute interpreter paths. A functioning
offline guest Python/rasterio/GDAL runtime has yet to be demonstrated.

Microsoft documents disabled networking and read-only mapped folders. Networking
defaults to enabled. Writable mappings can persist changes on the host. A later
approved configuration must explicitly disable networking and tightly scope
mappings. [Windows Sandbox configuration](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-configure-using-wsb-file)

The practical candidate is an explicitly authorized synthetic Sandbox smoke with
networking disabled, a read-only runtime/code/synthetic-input bundle, disabled
vGPU and a fresh controlled export location. The runtime bundle needs verified
hashes and a tested guest interpreter/GDAL startup. The guest worker must retain
the existing Job limits and publication checks. Guest memory configuration alone
does not establish the worker Job cap. Host export needs byte/hash validation
and accounting for staging, repeats, partials and reports. These are next-run
engineering conditions. No Sandbox configuration or private mapping is launched
or created here. [Microsoft sample configuration](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-sample-configuration)

Existing Domain/Private/Public firewall profiles reported enabled and raw default
outbound action 0. That state supplies no demonstrated interpreter/native-GDAL
outbound guarantee. Per-process policy, startup coverage, rollback and verification
would require separate authorized testing. No firewall or network settings were
changed. The Sandbox candidate is the recommended bounded path to test first.
Real processing remains blocked by the unproved OS/native offline boundary.

## Retained resource and data safeguards

| Item | Existing enforced synthetic limit |
|---|---|
| Windows Job memory / active process | 2 GiB / 1 |
| Deadline | 1,800 seconds |
| Cumulative writes / reports | 64 MiB / 8 MiB |
| Encoded product | 2 MiB |
| Initial free disk | 256 MiB |
| Network for a future real run | 0 bytes / 0 requests |
| Products / repeat check | 16 products per pass, 32 across two passes |

The B5 synthetic worker, Job assignment, reservation accounting, exclusive
publication, readback and determinism tests remain intact. New readiness code
only checks the proposed limits against those constants. It creates no staging,
products, reports or runtime resources. Actual isolation, guest limits and
export accounting must be demonstrated before a real adapter is enabled.

## Minimal next decisions and execution gates

1. Obtain genuine G3 finite-scope and G4 operational whole-tile decisions for
   these four and present use, with human reference and date.
2. Separately authorize and demonstrate the small synthetic OS-isolation/runtime
   smoke, including startup/native networking and controlled output export.
3. Bind a final four-image execution plan and human approval to actual package,
   catalog, receipt, code/runtime and original source hashes, new output paths
   and the unchanged cumulative resource limits. Recheck originals only then.
4. Enable only the minimal real adapter after those gates. Verify actual output
   grids, saved float32 fraction thresholding, readback and two-pass SHA equality,
   followed by R1a full preflight and independent product review.

G5/G6 execution and Formal R1 acceptance remain separate future decisions. No
new coverage threshold, mask-aware loss, patch rescue or contract relaxation is
introduced by this package.

## Validation on 2026-11-11

The final full regression run completed with **503 passed, 1 skipped, 3,994
warnings in 221.76 seconds**. The existing skip requires Windows symlink
privileges. The focused new/B5 run completed with **90 passed, 200 warnings in
14.16 seconds**. The full run includes all 42 new evidence-only cases and the
unchanged 48 B5 tests, including real Windows Job enforcement against synthetic
workers, storage ceilings, interrupted publication and two-pass product hashes.

The read-only local receipt check returned exit 0 with all five expected hashes,
`execution_enabled=false`, `source_bytes_checked=false`, G3/G4 PENDING and all
150 Formal R1 HOLD. Those receipt hashes matched before and after the work.
Existing processing code, R1a contract and historical manifests have no diff.
The original image/archive bytes were not accessed in this task, so this check
does not refresh their historical authenticity measurements. Real-product,
Sandbox/network-boundary and guest-runtime validation remain unperformed.
