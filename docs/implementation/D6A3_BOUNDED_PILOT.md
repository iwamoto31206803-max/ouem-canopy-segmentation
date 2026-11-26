# D6-A3-2 — Bounded OAM-TCD research patch pilot

[Issue #113](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/113) and [Draft PR #115](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/115) cover implementation and review. The integrated review approved the implementation and the owner subsequently authorized one attempt on four fixed OAM-TCD sources. The real attempt and reader checks passed. The private plan and receipts were reconciled locally; they were outside the independent implementation review.

The preceding integrated review recorded a Windows baseline of 768 passed and 2 skipped at the prior PR head. Those results and the earlier Val QA remain prior evidence.

## Inputs and admission

- Keep the frozen 2 Train / 2 fresh Val, with one A1-selected 224×224 window per source in full-grid coordinates. Reuse completed A1/A2 evidence. Preserve exact source IDs, folds, crop locations and A1 results. The implementation caps execution at two sources per split.
- The reviewed historical freeze and its exact v5/Val18 subindex have independent SHA pins in the code. Validate all 88 historical memberships against the pinned catalog, including aliases and envelopes. Historical Train reuse is permitted only in the same role and is recorded in receipts. Fixed Val18 and official Test stay separate. Envelopes and the 1 km proximity guard describe known catalog geography. GT54 is reserved for future external evaluation and its geographical independence remains unverified.
- Keep external-expected-SHA and local-observed SHA separate. Up to three sources may use the local-observed class for this pilot. Both classes require pinned provenance, applicable source/dataset/annotation rights evidence, and current input byte matches. Local-observed hashes support reproducibility and carry no independent publisher authentication. The writer decodes and rechecks the selected window against outer-connected exact-black support, without choosing a new crop or applying color-based NoData to white pixels.
- The approved plan binds exact A1 contracts, reports, candidate/result membership and allowlists, catalog and per-source source/crop/grid fields. Complete the private plan with `storage_policy: BOUNDED_WRITES_AND_OBSERVED_ALLOCATION`, exact `locations` (cache, evidence, output), `protected_roots`, and each selected source's `output_shape` and `crop_transform` from its pinned report. Keep the frozen proposal as historical evidence. Retain the final plan and approval SHA separately before execution and supply those expected pins to the CLI. The reviewer covered the implementation, the owner authorized the scope, and Codex reconciled the private pins locally.

## Resource limits and recovery

| Resource | Pilot limits |
| --- | --- |
| Windows Job | One process, at most 2,560 MiB committed; 2 GiB peak working-set stop; at least 6 GiB host memory available at start |
| Time | At most 120 s per source transaction and 600 s for the supervised attempt, from child launch through preflight, commits and publication. Job close/kill and bounded reaping follow a stop. |
| Output | At most 4 MiB staging per source and 16 MiB logical output. Maximum 24 files including transaction bookkeeping. At least 256 MiB free throughout, plus logical write headroom at start. |
| Observed allocation | At most 64 MiB of measured regular-file allocation on a Windows local fixed drive. Measurement surrounds staged writes and publication. Infrastructure changes and a filesystem quota are unnecessary for this pilot. |

The working-set check samples the Windows peak counter. A source watchdog can terminate the child and leave the claim for reconciliation. Output allocation is sampled at transaction boundaries and excludes filesystem metadata and transient OS effects. It offers neither a hard quota nor a power-loss guarantee. Local-only paths, no network fetches, and offline worker settings cover the normal CLI; the Python guard is not a host firewall or an arbitrary-code sandbox.

Every source is exclusively staged, read back and pinned before publication. Complete sources can be reverified and skipped using the same plan, approval and output, with `--resume`. An orphan claim, unverified partial, unexpected path, invalid digest, or interrupted manifest link/unlink pair requires manual reconciliation. The CLI preserves these files and does not automatically retry. After a forced stop inspect retained files before relying on any outputs. Reader opens a complete generation only against an independently retained manifest SHA and checks product digests, crop/grid and mask consistency. It uses the existing research-only training interface. Formal R1 remains HOLD, and D6-B is deferred.

## Reviewed Windows invocation

Independent review covered the implementation. The owner authorized the exact four-source scope, and Codex reconciled the private plan and evidence pins locally. Real runs use the existing venv and independently retained plan/approval SHAs. The following placeholders are intentionally non-executable.

```powershell
.\.venv\Scripts\python.exe -B scripts/d6a3_produce.py `
  --plan "<authorized private plan>" --plan-sha256 "<independently retained plan SHA>" `
  --root "<bound cache>" --evidence "<bound STAC directory>" --output "<bound fresh external output>" `
  --real-approval "<separately authorized token>" --real-approval-sha256 "<independently retained token SHA>"
```

Capture the printed manifest SHA and resource readings. Validate the output with `D6Data(output, manifest_sha256)` before any downstream use. Earlier synthetic integration exercised Windows controls on four 2048×2048 synthetic inputs.

## A3-2b real pilot result (2026-11-25)

**A3_2B_REAL_PILOT_PASS.** One reviewed-code attempt returned exit 0 and generated 2 Train and 2 fresh Val patches, K=1 per source. D6Data reopened all 12 TIFFs and returned all four normalized RGB and binary target tensors. Crop, affine, CRS, hashes, dtypes, ranges and the threshold of 0.5 on the saved Float32 fraction matched the contract. All selected source/STAC hashes and pinned catalog/A1/historical evidence matched again after execution. The source identities and products remain private. [Issue #113](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/113).

| Measurement | Observed |
| --- | ---: |
| CLI wall time / supervised attempt | 18.78 s / 14.36 s |
| D6Data construction and patch checks | 0.20 s |
| Peak working set / Job commit | 802,910,208 B / 780,427,264 B |
| Logical generation / TIFF bytes | 595,961 B / 578,917 B |
| Peak observed regular-file allocation | 627,648 B |

## Historical next-batch proposal

The bounded follow-up is recorded in [D6-A4 cached generation](D6A4_CACHED_GENERATION.md).

Use cached official Train inputs. A filename presence check found both originals for all 3,341 training-fold and 828 fold-0 catalog rows. The prior documentary inventory found 729 Train and 175 Val rows passing its metadata screen; the later fresh-Val screen recorded 40 known-clear rows before selecting six for QA. These are candidate counts. Accepted yield depends on rights, identity, independence, geometry and input bytes.

1. Freeze a distinct source-balanced roster by source group and region using existing metadata and rights receipts. Preserve official folds and paired source/envelope exclusion across every shard and historical benchmark. Start with one serial shard of about 25 sources, then use shards of 25–50 toward about 200 sources if the measured eligible pool supports it. Set the fresh-Val target from available independent groups, with roughly 20–30 as a planning range. Keep unresolved candidates on HOLD. Avoid repeated A1 QA where the report and source pins are reusable.
2. Extend only the reviewed four-source CLI cardinality and associated batch budgets in a separate change. Reuse A1 crop selection, K=1, source receipt/commit, Windows Job, manifest and D6Data readback. Keep a shared cross-shard identity check and bind each frozen shard to the common roster. Use fresh local roots, resume only verified states, and stop on integrity, resource or rights conflicts.
3. At the four-source average, 200 sources would occupy about 30 MB logical output and the measured throughput would imply roughly 16 minutes of producer wall time. These are sizing references from a small pilot that includes fixed catalog overhead and data-dependent compression. Budget from the existing 4 MiB per-source stage reservation plus batch bookkeeping (about 200 MiB for 50 sources before bookkeeping) with explicit free-space headroom. Check actual source distribution, RSS, wall time, bytes and eligibility yield on the first shard before setting final shard size and total run budget.

No larger batch was run. Formal R1 remains HOLD for all 150, and GPU training and D6-B remain deferred.
