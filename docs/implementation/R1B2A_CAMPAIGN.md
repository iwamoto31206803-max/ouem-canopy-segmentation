# R1b-2a — Persistent campaign safety contract

Safety implementation and offline synthetic tests only. No HTTP client, production
QA wrapper, actual campaign, 25-image QA, preview/GeoTIFF generation or GPU run was
executed. Historical G0–G3-A and Zenodo10 R1 HOLD remain unchanged. A campaign's
`execution_gate=OFFLINE_SAFETY_ONLY_NOT_REAL_DATA_APPROVAL` is not approval of
the proposed limits or any image's rights, validity or training eligibility.

## Fixed inputs and registry

`scripts/r1b_campaign.py` uses schema `ouem-r1b-campaign-v1`. Initialization:

* Hash the supplied plan bytes, require the caller's expected SHA, and recompute
  its canonical R1b-1 plan ID. An optional expected plan ID is also checked.
  No specific plan ID or image IDs are hardcoded.
* Require the complete first **25** IDs, matching the pool's initial prefix,
  official Train/fold roles and 20 Train/5 Val quota. No other pool ID is permitted
  by the operation API. Verify both local Zenodo metadata SHA pins.
* Freeze plan ID/file SHA, targets/splits, metadata hashes, cache location, exact
  25 TIFF/PNG relative filenames, verification scope, schema, campaign ID/time,
  limits and registry binding in `contract.json`. Source image bytes are explicitly
  `NOT_VERIFIED_R1B2B_REQUIRED`; only the metadata is read in this phase.
* Bind a permanent registry entry to **generation + target IDs/splits**, independent
  of campaign name, plan formatting, cache copy path or budget. An existing claim,
  including an interrupted initialization, prevents recreating the same cohort
  with a fresh root/ID to reset counters. The registry anchors the contract SHA and
  resolved root. Resume rechecks all these pins; no budget override is available.

Every mutating API also checks the current owner-lock hash, unchanged in-memory
and on-disk contract bytes, and equality with the last journaled state. Changing
limits/counters in memory cannot silently authorize I/O. A changed/missing owner
lock blocks operations and the old context refuses to remove another owner's lock.

Use one permanently retained, backed-up registry for this project. Changing/deleting
the registry, rolling back both registry and journal, or editing private Python
state can defeat bookkeeping; this is not an adversarial access-control system.
Never choose a second registry to evade a claim. A genuinely different cohort or
registry migration requires separate review and cumulative-budget reconciliation.
Moving a live campaign/root is not supported; use its fixed paths on resume.
Additional legacy dataset/checkpoint/audit roots can be passed with
`protected_roots=[...]` or repeatable CLI `--protect-root`; they are frozen in the
contract and both campaign/registry destinations must be disjoint from them.

## Journal and states

```text
<campaign>/contract.json
<campaign>/events/00000001.json ...   # immutable operation deltas + hash chain
<campaign>/campaign.lock             # exclusive owner PID + nonce
<campaign>/artifacts/<op>.partial     # reserved, incomplete output
<campaign>/artifacts/...             # published artifacts
<campaign>/recovered-locks/<sha>.json # retained, hash-checked old lock evidence
<registry>/<cohort-binding>/anchor.json
<registry>/<cohort-binding>/head.json
```

Each event records sequence, prior SHA, contract SHA, UTC, reason, changed operation
records and fatal state. Existing events are never overwritten. Exclusive event
creation and file `flush/fsync` precede an atomic `os.replace` of the separately
anchored head (the new head is itself flushed first). Operation deltas avoid
quadratic full-state snapshots consuming the proposed disk budget. Replay verifies
the whole chain/head and all completed/retained file sizes and SHA. A torn event,
extra event, missing tail, orphan `head.pending`, corrupt head or unknown file
fails closed; valid earlier records are retained. Automatic repair of journal/
publication corruption is **not implemented**. Preserve the tree for review.

Operations progress through RESERVED → receive/write progress → WRITTEN → SUCCESS.
An I/O error produces INTERRUPTED or leaves the earlier durable pending reservation
if the journal itself cannot be updated. Inspect reports STOP and partial file
paths/sizes/SHA. Explicit operator reconciliation records RECONCILED_FULL_CHARGE,
not success; partial artifacts stay retained and counted. Operation IDs are never
reused. SUCCESS here means bounded I/O/state publication, **not image QA PASS**.

## Reservations, bytes and stopping

Default maximum proposals: metadata transfer 2MiB, 25 charged requests, response
64KiB, storage 94MiB. API initialization may explicitly lower, never increase,
these limits. They are frozen by the anchored contract. Image/annotation HTTP and
Range caching have no operation kind; network bytes are permitted only for `stac`.

Before any managed operation, reserve network bytes, one charged request for STAC,
and **twice** the maximum output bytes (temporary + exclusive-publication window).
Check cumulative network/request caps, actual stored files, pending storage holds
and OS free space. Budget rejection occurs before the reader/writer is called.
Journal/head/contract/registry/lock files and reports also count against storage;
65,536 bytes of ordinary-operation control headroom are retained. Head-publication
temporary bytes are checked before writing. Free space is checked again on I/O.
Both campaign and registry volumes are checked (conservatively including pending
space), so placing them on different drives does not bypass the free-space gate.

`usage()` reports:

* `network_charged_bytes`: exact settled consumption or full unresolved/reconciled
  reservation. It must be used for budget enforcement.
* `known_received_bytes`: bytes returned by the bounded reader; **a lower bound**
  when a read may have failed after hidden consumption. Never replace the charged
  counter with this lower number during recovery.
* `requests`: conservatively charged request slots, including failed/pending or
  never-started reservations. These are not proof that a real HTTP request occurred.
* `disk_bytes`: current logical file lengths, including controls and partials;
  `unresolved_storage_reservation_bytes` is reserved space not yet represented by
  files, and `storage_including_reservations_bytes` is their sum. There is no
  artifact cleanup/refund API. Unused storage reservation is released on settlement
  only after retained files have been measured. Network usage is cumulative.

The bound applies to application-delivered response-body bytes, not HTTP headers,
TLS, sockets' internal buffering or total wire traffic. A reader must honor the
requested remaining-byte limit. Over-return is logged at the observed larger
amount and becomes fatal (no normal recovery). EOF must occur below the response
allowance; exhausting it without verified EOF is STOP, not a truncated success.
No probing byte beyond the allowance is requested. Unknown read consumption keeps
the **entire reservation**, even across process exit. Failed attempts/retries use
new operation IDs and the same cumulative budget; nothing automatically retries.

File capacity is logical byte accounting plus OS free-space checks, not exact
filesystem cluster/metadata allocation or a measured hardware peak. Full batch
peak, library memory, Drive/SMB semantics and power-loss behavior are unmeasured.
Use a local filesystem supporting exclusive create, hard links, fsync and atomic
rename (tested here on Windows). Unsupported publication fails without overwrite;
there is no unsafe fallback. File hashes detect post-pause changes on resume.

## Minimal R1b-2b API

```python
from r1b_campaign import Campaign

# Initialization is separate; no data run is approved by creating this contract.
Campaign.create(root, registry, plan_path, expected_plan_sha, cache_root, "r1b2-pilot")

with Campaign.open(root, registry, plan_path, expected_plan_sha,
                   expected_id=reviewed_plan_id) as campaign:
    campaign.reserve("preview-1638", 1638, "train", "preview",
                     "artifacts/1638/preview.png", max_output_bytes=600_000)
    # Produce encoded bytes in bounded memory, not via an unmanaged disk writer.
    campaign.write_bytes("preview-1638", encoded_preview_bytes)
    campaign.commit("preview-1638")
```

`reserve(op, image_id, split, kind, path, max_output_bytes, network_bytes=0)`
supports `stac`, `preview`, `geotiff`, `report`, `temporary`. Paths must stay in the
fresh external campaign's artifacts directory. Existing outputs, traversal, links,
Windows reparse points/DOS device names, case-folded duplicate and partial-path
collisions are rejected. Data/cache/repository destinations are forbidden.
`write_bytes()` is exclusive and length-bounded before write; it does not expose
an unchecked output path to a native encoder. R1b-2b can use Pillow BytesIO or
rasterio MemoryFile, then pass their encoded bytes through the same budget. Native
writers saving to arbitrary temp files would bypass this API and are not approved.

For STAC, **reserve before opening the HTTP request**, pass a lazy bounded reader
to `receive()`, then `write_bytes()` and `commit()`. No actual HTTP adapter is
provided. R1b-2b must implement the official item-ID URL allowlist, no hidden
retries/redirect requests, streaming/encoding/body checks, and STAC JSON/provenance
validation. A `stac` label alone cannot identify an arbitrary caller's network
content; never label image/Parquet acquisition as STAC. Faults stop the campaign.

Old M1-5B's finally-only network writeback and uncontrolled encoder writes cannot
be used alongside a second independent budget. Its source/license/separation/QA
functions and M1-5D preprocessing computation remain reusable **without changing
old gates**; R1b-2b must route their I/O through this ledger. No old code is edited.

## Windows initialization, inspection and resume

These are instructions for a separately reviewed offline initialization; no real
campaign was initialized for this PR. Use fresh external `$Root`, a permanent
external `$Registry`, and existing read-only `$Cache`. Keep registry/campaign
disjoint and outside the repo/cache. Do not commit their local contracts or STACs.

```powershell
$Plan = "docs/implementation/R1B1_CANDIDATE_PLAN.json"
$PlanSHA = (Get-FileHash $Plan -Algorithm SHA256).Hash.ToLowerInvariant()
$PlanID = (Get-Content $Plan -Raw | ConvertFrom-Json).plan_id
.\.venv\Scripts\python.exe scripts/r1b_campaign.py init `
  --root $Root --registry $Registry --plan $Plan --plan-sha256 $PlanSHA `
  --plan-id $PlanID --cache $Cache --campaign-id r1b2-pilot `
  --protect-root $LegacyDataRoot --protect-root $LegacyCheckpointRoot
.\.venv\Scripts\python.exe scripts/r1b_campaign.py inspect `
  --root $Root --registry $Registry --plan $Plan --plan-sha256 $PlanSHA --plan-id $PlanID
.\.venv\Scripts\python.exe scripts/r1b_campaign.py resume-check `
  --root $Root --registry $Registry --plan $Plan --plan-sha256 $PlanSHA --plan-id $PlanID
```

Normal context exit with no pending operation removes only that context's lock.
Reopening rehashes files and input pins; it does not recreate a campaign. A pending
normal exit raises STOP and keeps its lock. On an exception the owned lock may be
released, but unknown reservations remain and still block resume. A crash leaves
the lock. Inspect is read-only and never deletes lock/temporary files.

For abnormal recovery: stop/confirm all old owners yourself; preserve inspection
output externally; inspect the original lock SHA and artifacts. Only after that,
use `reconcile --acknowledge-stopped-owner` with all the same arguments above and
`--lock-sha256 $ObservedLockSHA` **if** a lock exists. If no lock exists, omit the
lock SHA. A mismatched lock or absent acknowledgement refuses recovery. The old
lock is archived with its SHA; an exclusive recovery guard serializes recovery.
Pending operations become failed/full-network-charge records; originals/partials
are never deleted and the old operation cannot be executed again. Then run
`resume-check`. Reconciliation is not a retry or proof of completed output.

The acknowledgement is an operator responsibility, not an automatic proof that a
PID is dead. Never reconcile a live owner. A stale recovery guard, journal damage,
fatal overrun or initialization failure requires separate review; do not delete
locks/claims or edit counters to make the run open. A corrupt/unavailable contract
or file raises nonzero without reporting success. Inspect returns 2 for a readable
STOP state, 0 for CLEAN_PAUSED; this does not grant real-data execution approval.

## Verification and remaining gate

Synthetic tests cover pinned inputs, fixed 25/role rejection, reservations and
cumulative budgets, failed read/partial write, exact output hashes, clean resume,
separate-process `os._exit` with lock retention, explicit full-charge recovery,
double operation rejection, path protection, Windows hard-link publication,
simulated reparse attributes, journal/head/artifact corruption and orphan head
failure. No test performs actual network requests or source image processing.

### PR #44 independent-review hardening

`write_bytes()` checks the actual completed partial against the supplied bytes,
then journals `written_file={bytes, sha256}` with state WRITTEN. `commit()` accepts
only that persisted WRITTEN state; an externally supplied partial in RESERVED
does not qualify, and INTERRUPTED cannot publish. It rechecks the partial's size
and SHA against the recorded receipt before linking, and checks the published
link against that same receipt before settlement. Missing/changed bytes cause
STOP/INTERRUPTED with the receipt and existing fault/publication records retained.
Older WRITTEN records without a receipt fail closed; no receipt is reconstructed
from whatever file happens to be present. Cooperative filesystem ownership remains
required; this is not a guarantee against arbitrary concurrent external writers.

`inspect()` reports a retained `reconcile.lock` as STOP even if all operations are
settled and the campaign owner lock is absent. `stop_reasons` distinguishes
UNRESOLVED_OPERATIONS, CAMPAIGN_OWNER_LOCK_PRESENT, RECONCILIATION_LOCK_PRESENT and
FATAL_ACCOUNTING_STATE; `reconcile_lock_sha256` identifies the guard for review.
Inspection never deletes or repairs it. Recovery of that stale guard still
requires independent review, rather than silently declaring CLEAN_PAUSED.

A read-only check of the real R1b-1 plan ID/file SHA and both local Zenodo metadata
pins plus its 25 target roles passed. This is **input-contract verification only**:
no source pixels, real STAC retrieval, full-batch QA, outputs, peak or real campaign
execution were verified. Rights/category/validity and external separation remain
HOLD/NOT_VERIFIED for R1b-2b. No training acceptance changed.

Before R1b-2b: independently review this safety layer and the proposed limits;
implement the bounded HTTP and native-encoder adapters plus plan-bound QA wrapper;
measure actual temporary/retained bytes and stop **before** either cap is exceeded;
perform all 25 visual/source/rights/external-catalog reviews and source hashes;
build new R1a receipts/manifest and require R1a verify/preflight. Review measured
peak and campaign-journal growth rather than assuming 94MiB fits. No blanket
hardware power-loss guarantee, auto journal repair, arbitrary cleanup, registry
migration, paid resource, model or GPU execution is introduced.

R1b-2b must also specify the explicit execution-approval receipt/process that binds
the independently reviewed contract/plan/limits. Do not mutate the frozen offline
contract or interpret its creation as real-data approval.

Measured verification on Windows/Python 3.11.9, existing venv:

* Full suite: **323 passed, 1 skipped, 676 warnings (87.49s)**.
* Campaign-only synthetic tests: **34 passed (17.44s)**.
* The skip is the existing Windows real-symlink-creation limitation; this module's
  reparse guard is tested with simulated Windows file attributes. Actual local
  exclusive locks, hard-link publication and subprocess crash behavior were tested.
  Warnings are existing rasterio/Affine deprecations and the duplicate-ZIP test.
* CLI help/init/inspect/resume-check tests and `git diff --check` passed.
* Real plan/metadata **read-only input check PASS**, not real-data execution PASS.
  Plan ID remains `ouem-r1b1-fbb4ed018dc3cfded676d6ba09229ef7ccfb80fa7bef9b9b67a1df84864fcc5a`.
  This checkout's plan file SHA is
  `98abe4b98cb629c3c596f70718db2854a2df675f6c7781792823f33f6115fcbc`.
  Compute/pin actual bytes as shown above; Git line-ending conversion can differ
  from the original generated LF receipt without changing the canonical plan ID.
* G0 fixed manifest SHA remains
  `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca`.
  No historical scripts/data/manifests/checkpoints were modified. New dependencies:
  none; the existing G0 streaming SHA/serializer are reused.
