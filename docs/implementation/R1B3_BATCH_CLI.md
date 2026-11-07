# R1b-3 batch CLI (batches 02–05)

Purpose: reduce repetitive Codex/PowerShell work. The CLI is a thin, offline
operator wrapper around the **existing** `r1b3_scope.py` and QA Runner.
It is **not** an independent approval issuer or a data-admission tool.

The immutable Campaign/scope from Issues #55–57 stays untouched. Batch 01 is
already completed and is not accepted by this helper. The original 25-image
Campaign is never reprocessed.

## Prerequisites

- Merge the reviewed PR, then fast-forward the GEEKOM-A6 checkout to a **clean
  main**. The current commit will differ from PR #54's historical commit.
- Use the existing verified external `r1b3-pool150-scope-01.json`, Campaign
  and Registry. Do **not** initialize or reset them again.
- Run within the existing Windows Python virtual environment, or use
  `.\.venv\Scripts\python.exe` as shown.
- By default, paths are those used by the existing GEEKOM-A6 Campaign under
  `C:\dev\ouem-data-audits` and cache `C:\dev\ouem-data-cache\oam-tcd-zenodo-v1`.
  `OUEM_R1B3_AUDITS` and `OUEM_R1B3_CACHE` override these locations.
  The reviewed scope SHA, contract SHA and Batch-01 template SHA are pinned.

## 1. Read-only inspection

```powershell
cd C:\dev\ouem-canopy-segmentation
.\.venv\Scripts\python.exe -B scripts\r1b3_batch.py inspect --batch 2
```

Inspection refuses dirty Git, any branch other than main, changed scope or
contract, unclean Campaign, already-processed target IDs, incomplete old evidence,
nonzero network usage, or exhausted budget. The existing scope verifier also
reconstructs its references read-only.

## 2. Prepare an **unapproved** candidate

```powershell
.\.venv\Scripts\python.exe -B scripts\r1b3_batch.py prepare --batch 2
```

This is the **only** phase that creates an output file, via exclusive create.
The default output is
`C:\dev\ouem-data-audits\r1b3-batch02-local-01-approval-candidate.json`.
It does not overwrite a previous candidate. The command reads and hashes the
50 existing TIFF/PNG originals (without pixel decoding), pins the current
clean Git environment, and prints the **candidate SHA-256**.

The result is `approved:false`, `reviewer_ref=PENDING_INDEPENDENT_REVIEW`;
the runner cannot execute it. Submit its contents/SHA and the inspect output
for **independent review before proceeding**.

## 3. Independent authorization (outside this CLI)

After the exact candidate has been reviewed, **preserve the candidate as-is**.
An operator may separately create a *new* reviewed Approval JSON with exactly
these four field changes:

- `approved` = `true`
- `approval_id` = `r1b3-batch02-local-01-approved-v1`
- `reviewer_ref` = the actual external review reference, not PENDING
- `execution_intent.authorization_status` = `APPROVED_FOR_BATCH02_LOCAL_QA`

Use the matching batch number for 03–05. This CLI intentionally offers **no**
`approve` or `seal` command. Review and approval issuance must remain separate
from the automatic preparation process. Record the approved file's SHA-256.
All other fields must be byte-semantically identical to the reviewed candidate.

## 4. Validate approved inputs (read-only)

Replace the three placeholders with the independently checked values:

```powershell
.\.venv\Scripts\python.exe -B scripts\r1b3_batch.py check --batch 2 `
  --candidate-sha256 "PASTE_REVIEWED_CANDIDATE_SHA256" `
  --approval "C:\dev\ouem-data-audits\r1b3-batch02-local-01-approval-approved.json" `
  --approval-sha256 "PASTE_REVIEWED_APPROVAL_SHA256"
```

In addition to the existing Runner's check-approval, the wrapper ensures that
the candidate and approved document differ **only in the four authorized
fields**, and rehashes all 50 current source files. Unknown scope requests,
unexpected run IDs, incorrect environment/COCO and changed source SHA fail closed.

## 5. Execute exactly one approved batch

Only following a separate explicit execution authorization:

```powershell
.\.venv\Scripts\python.exe -B scripts\r1b3_batch.py execute --batch 2 `
  --candidate-sha256 "PASTE_REVIEWED_CANDIDATE_SHA256" `
  --approval "C:\dev\ouem-data-audits\r1b3-batch02-local-01-approval-approved.json" `
  --approval-sha256 "PASTE_REVIEWED_APPROVAL_SHA256" --execute
$QaExit = $LASTEXITCODE
"QA native exit=$QaExit"
```

The command invokes the existing `r1b3_scope` Campaign and QA Runner with
the **single** reviewed batch's 25 IDs and fixed run ID. The explicit
`--execute` switch is mandatory. There is no multi-batch loop, retries,
network/STAC fetch, GeoTIFF creation, Registry reset or automatic acceptance.

A native exit of `2` can indicate R1 HOLD; inspect the Campaign and
per-image reports. Do not retry an already-started run or clear locks/counters.

## Other batches

After each completed batch has been independently reviewed, use the same
commands with `--batch 3`, `4`, or `5`. There is no new code change
needed solely to advance the batch number.

## Verification boundaries

Synthetic unit tests cover candidate creation, refusal to overwrite or write
inside protected roots, four-field review matching, rehash failure on changed
sources, and single-batch dispatch. Run:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests\test_r1b3_batch.py -q
.\.venv\Scripts\python.exe -B -m pytest -q
```

This PR does not produce real candidate JSON or QA outputs. The rights/source,
visual, segmentation, geographic-independence and R1a admission gates remain
separate and HOLD until independently resolved.
