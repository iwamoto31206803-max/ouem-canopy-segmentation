# R1d-1B-5 bounded four-image conversion rehearsal

The implementation baseline is main at
`8be453ebe9535a6d16160368b2e00d128403ddbc`. The task is
[Issue #60](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/60).

## Available entrypoint

`scripts/r1d_bounded_prepare.py` runs a plan-pinned synthetic rehearsal for
Train 717/1054 and Val 1241/461. It generates tiny synthetic sources inside a
fresh temporary output tree. The plan has a closed set of fields, source and
evidence hashes, exact four-ID order, splits and resource limits.
User-specified source paths are rejected.

The real-mode gate returns `REAL_EXECUTION_DISABLED` before original-image
access or output creation. Native-library network isolation is still
`NOT_VERIFIED`. Genuine real input/review/approval integration remains pending.
G3/G4 decisions and execution approval are still required.

The synthetic stub has exactly `kind=SYNTHETIC_TEST_ONLY` and the pinned plan
file SHA. It carries no human approval or G3/G4 decision. Reports retain
`formal_r1=HOLD`, `G3=PENDING`, `G4=PENDING`.

## Reused conversion and product contract

The existing Zenodo converter gains three keyword-only hooks.

- `product_writer` supplies a bounded writer.
- `preview=False` suppresses preview generation.
- `expected_shape` is checked immediately after `output_grid`.

Existing callers retain their default conversion and preview behavior.
The rehearsal uses existing semantic union, validity checks, local-ground grid,
area reduction, RGB rounding, fraction threshold and saved readback.
All-zero/all-one masks are retained. Binary masks use the stored float32
fraction at threshold 0.5.

Four products are generated per image per pass, followed by an independent
repeat tree. The 32 output GeoTIFF hashes must match in corresponding pairs.
The eight tiny TIFF/PNG source fixtures and four source STAC fixtures are
also created within the same budgeted tree. Real pilot shapes are recorded
as 390x390, 391x391, 394x394 and 337x337 respectively. Synthetic fixtures use
4x4 outputs. Every executable fixture is explicitly synthetic.

## Enforced resources

| Resource | Ceiling or floor |
| --- | --- |
| Cumulative newly written file content | 64 MiB ceiling |
| Reports/evidence/control content within that budget | 8 MiB ceiling |
| Encoded product before disk publication | 2 MiB ceiling |
| Worker process and Job commit | 2 GiB ceiling |
| Active processes in worker Job | 1 |
| Worker deadline | 1,800 seconds |
| Free disk before output creation | 256 MiB floor |

Limits can be lowered for failure tests, while the free-disk floor is fixed.
One reservation accounts payload bytes plus 1,024 bytes for two journal records.
This conservative control allowance is never refunded. Failed payload
reservations remain fully charged.

The append-only journal is flushed before payload creation. Payloads are
written exclusively to retained partial files, flushed, rehashed and published
using an exclusive hard link. An existing destination causes failure. Completed
payloads are retained on later failure. Paths and ancestors are checked for
symlinks/reparse points. Run output is forbidden within the repository, existing
data/audit caches and model cache.

`retained()` is controller-only reconciliation after worker exit. It checks
committed size/SHA and reports unresolved reservations. A corrupt journal stops
reconciliation. A fresh invocation rejects the existing root. Processing resume
and budget reset require later implementation and authorization.

The budget measures file-content writes and conservatively reserved control
bytes. Filesystem allocation/NTFS metadata are outside that measurement.
Current retained file bytes and charged upper bounds are reported.

## Windows hard-limit verification

A trusted base-Python bootstrap waits on stdin while the controller attaches it
to a Job Object. The controller installs and queries process/job memory ceilings,
an active-process limit of one, kill-on-close and verified process membership.
Then the worker is released. The worker also checks its enclosing Job.
Windows venv redirectors are avoided by launching the base interpreter directly
and supplying the existing import paths after assignment.

The controller enforces the remaining elapsed deadline and closes/kills the Job
on interruption. Unsupported platforms and failed Job installation or membership
checks stop execution. Python bytecode writes and GDAL PAM sidecar creation are
disabled for the worker.

Python audit hooks deny socket, urllib and process-launch calls during worker
processing. Tests also exercise conversion with network APIs replaced by
failure functions. Native code and interpreter/import startup are outside the
Python-level audit guarantee. OS-wide network isolation remains unverified,
and real-data execution is disabled at an unconditional mode gate.

The implementation assumes an ordinary local filesystem and a trusted local
operator. It does not establish an adversarial filesystem sandbox against a
separate process racing directory replacement. Intended output files are
created beneath a newly owned tree and checked before publication.

## Synthetic Windows example

Use a fresh PowerShell session and a temporary folder outside the repository.

```powershell
$Demo = Join-Path $env:TEMP ("ouem-r1d-synthetic-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $Demo | Out-Null
$env:OUEM_SYNTHETIC_DEMO = $Demo
@'
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import r1d_bounded_prepare as bounded
root = Path(os.environ["OUEM_SYNTHETIC_DEMO"])
plan_bytes = bounded.encoded(bounded.synthetic_plan())
(root / "plan.json").write_bytes(plan_bytes)
stub = dict(kind="SYNTHETIC_TEST_ONLY",
            plan_sha256=bounded.digest(plan_bytes))
(root / "synthetic_stub.json").write_bytes(bounded.encoded(stub))
'@ | .\.venv\Scripts\python.exe -B -
$PlanSHA = (Get-FileHash "$Demo\plan.json").Hash.ToLowerInvariant()
$StubSHA = (Get-FileHash "$Demo\synthetic_stub.json").Hash.ToLowerInvariant()
.\.venv\Scripts\python.exe -B scripts\r1d_bounded_prepare.py `
  --plan "$Demo\plan.json" --plan-sha256 $PlanSHA `
  --stub "$Demo\synthetic_stub.json" --stub-sha256 $StubSHA `
  --output "$Demo\run"
$RunExit = $LASTEXITCODE
```

Exit zero requires the isolated worker and complete four-image repeat result.
`execution.json` is the controller outcome. `result.json` records worker progress
and failure details when the report reservation succeeds. `journal.jsonl` retains
cumulative reservations/commits. If a limit prevents a terminal report, the CLI
returns failure and preserves journal and partial files. A missing report cannot
grant success.

## Tests and review boundary

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests\test_r1d_bounded_prepare.py
.\.venv\Scripts\python.exe -B -m pytest -q
```

Tests exercise four-ID conversion/readback/repeat bytes, uniform labels, strict
plan and stub binding, source/evidence hashes, paths, rights/token/shape rejection,
encoded/write/report budgets, partial retention, duplicate output, corruption,
free disk, fail-fast and real-mode refusal through Python and CLI.
Windows tests actually provoke memory allocation refusal, process-creation
refusal, deadline termination and a denied Python socket call, and execute the
isolated synthetic worker. Non-Windows systems exercise fail-closed behavior
and skip actual Windows enforcement tests.

This PR implements synthetic safety mechanics. The real pipeline still requires
verified native/offline network isolation, genuine real-plan/source/review binding,
the separate G3/G4 decisions and an approved execution contract.
R1a contract and final `r1_review` follow the existing review process.
Real-image G5/G6 products, source staging and final R1 preflight were not run.

## Local validation snapshot

On Windows/Python 3.11.9, the final full suite completed with **461 passed,
1 skipped, 3,994 warnings** in 238.19 seconds. The new safety test file completed
with **48 passed, 200 warnings** in 11.34 seconds. An earlier focused
converter/R1a/safety run completed with 148 passed, 1 skipped and 427 warnings
before the last two strict-type cases were added.
The existing skip concerns unavailable Windows symlink-creation privilege.
Warnings are existing rasterio/Affine deprecations and the intentional duplicate
ZIP fixture.

An additional isolated synthetic run completed with 32 matching product files.
The queried hard Job ceiling was 2,147,483,648 bytes, peak Job commit was
90,599,424 bytes and worker elapsed time was 2.922 seconds. Final retained
content was 52,925 bytes, conservative cumulative charge was 87,717 bytes and
unresolved operations were zero. These measurements describe tiny synthetic
fixtures. The memory-exhaustion test separately exercises a 64 MiB hard ceiling.

That run's temporary controller report SHA-256 was
`073728a778bda98d34839ddecb83fb53c0185fd30b03a555e4d9e9043733c1e8`.
Reports include timing/resource observations, so their bytes can vary between
runs. Product-byte determinism is checked within each independent repeat.
