# R1b-3a: fixed 125-image local QA scope

Implementation only. No real scope file, Campaign, Registry, Approval, image QA,
network request or derived product is created by this PR. R1 acceptance remains
HOLD. Issue #53 is the parent plan; this document does not approve execution.

## Versioned scope and historical compatibility

`scripts/r1b3_scope.py` adds `ouem-r1b3-scope-v1` and
`ouem-r1b3-local-campaign-v1`. It subclasses the existing Campaign and reuses
its journal, claim, lock, reservations, bounded writes, exclusive publication,
inspect and explicit reconciliation. Small input/contract hooks default to the
unchanged v1 behavior. `r1b_campaign.py` still accepts only the initial 25 and
positive limits no higher than its original 94 MiB / 2 MiB / 25-request proposal.
Old contracts, files and journal bytes are not migrated or rewritten.
Historical Approvals are retained unchanged. Because Git/Runner fingerprints are
commit-bound, executing any newly reviewed code still requires a fresh matching
Approval; compatibility does not silently authorize an old approval on a new commit.

The fixed parent is `docs/implementation/R1B1_CANDIDATE_PLAN.json`:

```text
plan_id: ouem-r1b1-fbb4ed018dc3cfded676d6ba09229ef7ccfb80fa7bef9b9b67a1df84864fcc5a
Windows checkout SHA: 98abe4b98cb629c3c596f70718db2854a2df675f6c7781792823f33f6115fcbc
```

The scope preserves the parent's ordered 150 unique images/sources, official
Train membership and four-Train/one-Val schedule. Only the exact suffix after
the initial 25 is executable: 125 images, Train100/Val25. No substitutes,
duplicates, missing/extra/reordered records or altered split/source names pass.
All 150 records are compared with pinned JSONL metadata (IDs, source identities,
fold, license, dimensions and transformed metadata bounds). Source aliases that
cross official split/fold are rejected. Test metadata is used for separation;
Test pixels are never inspected. The existing Runner retains its full metadata
group/footprint checks, including the reused 25 and known opposite-role sources.
Metadata separation does not certify independence from unlisted sources.

The scope binds absolute parent/cache/legacy locations and both metadata hashes.
Each executable record specifies exact TIFF/PNG relative filenames; existence,
cache containment and reparse/symlink guards are checked without pixel decoding.
New source hashes are recorded by QA; independent archive authentication is still
a separate evidence gate. No unknown rights are filled in automatically.

`scope_id = "ouem-r1b3-" + SHA256(encoded(scope_without_scope_id))`.
`encoded` is the existing sorted-key, ASCII-escaped, indented UTF-8 JSON with final
newline. A separate SHA pins actual scope file bytes. No creation timestamp is
added: repeated builds with the same paths/evidence are byte-identical. A path
change requires a new reviewed scope; this is not a portable dataset manifest.

## Old evidence is referenced, not accepted again

The builder requires the old v1 Campaign to be CLEAN_PAUSED. It verifies its
hash-chain, contract and committed report/preview receipts for all 25. Each
report's ID/split/generation and current original TIFF/PNG SHA are checked.
The scope records relative report/preview paths and SHA, source hashes, old
contract SHA and root/Registry references. Every scope validation rechecks these
references read-only; missing/changed evidence or unresolved old state stops.

No old image becomes a new target or a new SUCCESS operation. The explicit
`r1_acceptance=NOT_INHERITED` reference marker prevents interpreting old smoke/QA
results as R1 admission. References do not repair missing rights, category or
visual reviews. Old missing white/shape diagnostics remain missing; they are
not automatically filled by rerunning the old images.

## New Campaign and partial Approval

The new class has local-only limits: network=0 bytes, requests=0,
response cap <=64 KiB (unused), storage <=256 MiB. Lower storage limits are
supported; increasing beyond the proposal is rejected. STAC and GeoTIFF
reservations are explicitly forbidden. No valid mask, product generator or loss
change is introduced. Temporary/control/report bytes use the existing ledger
accounting, including unconfirmed reservations; all five batches share one budget.

The permanent claim is keyed by source generation and the exact 125 targets.
Changing campaign name/root/parent scope hash does not reset a claim in that
Registry. Registry location is a trusted operator binding, not a machine-global
anti-tamper service: using a different Registry to bypass accounting is prohibited.
Old Campaign/Registry, parent, scope, cache and repository are protected inputs;
also pass other historical audit/checkpoint/approval roots to `--protect-root`.

Use ordinary Approval v2 with exact contract/plan/environment binding, plus:

- `scope_id` and `scope_sha256` equal the new contract fields;
- `allowed_image_ids`: nonempty subset of a single fixed batch, at most25;
- scopes limited to qa_local / qa_preview / qa_report;
- approved:false until independent review and a separate explicit instruction.

Batch membership is consecutive blocks of25 in the fixed suffix (five batches,
each Train20/Val5). Cross-batch approvals and STAC/GeoTIFF scopes fail. IDs omitted
at execution use only the approved subset. Per-image operation names stay fixed,
so a new run ID cannot reprocess an image. After a crash, inspect/reconcile;
never reset claims, delete locks, change counters or retry the same operation.

## Coverage diagnostics are not technical or admission overrides

Only the new Campaign enables additional `coverage_review`. Existing black
component results are reused; exact-white components use the same four-connected
implementation. It records edge-connected black/white pixels, label intersections,
largest-five bbox/occupancy data and possible corner/wedge review candidates.
Even a sub-1% edge component can be flagged; no new acceptance threshold exists.
White buildings, dark real objects and a single dark edge pixel are not declared
NoData. Flags mean REVIEW_REQUIRED; no flag means NOT_VERIFIED, not visual PASS.
`no_data_classification=NOT_DETERMINED`, `valid_mask_generated=false` always.

The historical >=1% black technical HOLD remains separate. The new diagnostic
never changes technical_status, teachers, uniform masks or R1 acceptance. A clean
technical report can still require coverage review. Shape summaries are sampled
from the largest five components, so absence of a shape candidate is not proof
that no small wedge exists; edge totals still include all components.

## Windows procedure: non-executing instructions

Do not execute these production commands until scope/initialization/execution
approvals have been independently reviewed. All paths are operator arguments;
the implementation hardcodes no GEEKOM data location.

1. Pin a reviewed clean commit and parent Plan ID/file SHA. Read-only inspect the
   old Campaign, confirm source/evidence hashes, and choose fresh external paths.
2. Only when scope preparation is authorized, create the scope (new file only;
   no Campaign/Approval/QA is created):

```powershell
.\.venv\Scripts\python.exe -B scripts/r1b3_scope.py build-scope `
  --parent-plan $ParentPlan --parent-sha256 $ParentSHA --parent-plan-id $ParentID `
  --cache $Cache --legacy-root $OldCampaign --legacy-registry $OldRegistry `
  --output $NewScope
$ScopeSHA = (Get-FileHash $NewScope -Algorithm SHA256).Hash.ToLowerInvariant()
```

3. Independently review the scope content/SHA and 25 read-only evidence references.
   Campaign initialization is a separate authorization that creates a permanent
   claim. Only then use:

```powershell
.\.venv\Scripts\python.exe -B scripts/r1b3_scope.py init `
  --root $NewCampaign --registry $NewRegistry --cache $Cache `
  --plan $NewScope --plan-sha256 $ScopeSHA --campaign-id $NewCampaignID `
  --protect-root $OldApprovals --protect-root $CheckpointRoot
```

For this new entry point, existing `--plan/--plan-sha256` arguments refer to the
**scope file**, and optional `--plan-id` to its scope_id; the parent Plan is
separately pinned inside that file/contract. Do not pass the parent as the scope.

4. Bind a new approved:false Approval candidate to the new contract/environment,
   scope fields and first batch only. Obtain independent review; the CLI does
   not issue/activate approvals. After explicit execution authorization:

```powershell
.\.venv\Scripts\python.exe -B scripts/r1b3_scope.py check-approval @CommonArgs --ids @BatchIDs
$CheckExit = $LASTEXITCODE
if ($CheckExit -ne 0) { throw 'STOP: approval failed' }
# Only after CLEAN_PAUSED inspection and explicit execution approval:
.\.venv\Scripts\python.exe -B scripts/r1b3_scope.py run --execute @CommonArgs --run-id $FreshRunID --ids @BatchIDs
$QaExit = $LASTEXITCODE
Write-Output "QA native exit=$QaExit"
```

`CommonArgs` binds new root/Registry/scope/SHA and reviewed Approval/SHA. Never add
fetch-missing-stac. Inspect after success/failure using this same new entry point.
Native2 indicates R1 HOLD, not automatic failure; per-image FAIL may be recorded
before later approved images run, while unsafe reservations/encoder errors abort.
Review each batch before approving the next; no automatic five-batch loop exists.

## Verification boundary

Tests use synthetic150 metadata/rasters and temporary legacy/new Campaigns only.
One test reads the repository's real parent Plan to confirm125/100/25 derivation;
it does not read real source pixels/cache or create a production scope. Real125
QA, real new initialization, production Approval, capacity/RAM/time and subsequent
training eligibility remain NOT_VERIFIED/HOLD. No new dependencies are required.

Verification on Windows / Python3.11.9 / existing venv (2026-11-07):

- `python -B -m pytest -q`: **407 passed, 1 skipped, 3794 warnings, 385.32s**.
  The 11 new cases are included; existing tests all remain successful. The skip
  is the existing Windows real-symlink-creation limitation. Warnings are existing
  Affine deprecations and the deliberate duplicate-ZIP fixture.
- New CLI help/build-scope help, synthetic scope CLI creation/no-overwrite,
  check-approval guard and `git diff --check` passed.
- Real parent Plan was read only for ordered125/Train100/Val25 derivation.
  No real125 raw pixel, new scope output or new Campaign was created.
- Read-only inspection of the historical Campaign remains CLEAN_PAUSED with
  54 operations/163 events, 7,865,518 bytes, HTTP0, no unresolved reservations;
  its contract SHA remains
  `4c96dda2b42b775f77f89153aab2027c50808e933c70f0f80d3f64de9b5cda42`.
  The proposed production R1b-3 Campaign/Registry paths do not exist.
