# D6-A3-2 — Bounded OAM-TCD research patch pilot

[Issue #113](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/113) and [Draft PR #115](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/115) cover implementation and review. The approved execution scope for this work was synthetic. The private four-source proposal remains pending independent review and authorization before a real attempt.

The preceding integrated review recorded a Windows baseline of 768 passed and 2 skipped at the prior PR head. Those results and the earlier Val QA remain prior evidence.

## Inputs and admission

- Keep the frozen 2 Train / 2 fresh Val, with one A1-selected 224×224 window per source in full-grid coordinates. Reuse completed A1/A2 evidence. Preserve exact source IDs, folds, crop locations and A1 results. The implementation caps execution at two sources per split.
- The reviewed historical freeze and its exact v5/Val18 subindex have independent SHA pins in the code. Validate all 88 historical memberships against the pinned catalog, including aliases and envelopes. Historical Train reuse is permitted only in the same role and is recorded in receipts. Fixed Val18 and official Test stay separate. Envelopes and the 1 km proximity guard describe known catalog geography. GT54 is reserved for future external evaluation and its geographical independence remains unverified.
- Keep external-expected-SHA and local-observed SHA separate. Up to three sources may use the local-observed class for this pilot. Both classes require pinned provenance, applicable source/dataset/annotation rights evidence, and current input byte matches. Local-observed hashes support reproducibility and carry no independent publisher authentication. The writer decodes and rechecks the selected window against outer-connected exact-black support, without choosing a new crop or applying color-based NoData to white pixels.
- The approved plan binds exact A1 contracts, reports, candidate/result membership and allowlists, catalog and per-source source/crop/grid fields. Complete the private plan with `storage_policy: BOUNDED_WRITES_AND_OBSERVED_ALLOCATION`, exact `locations` (cache, evidence, output), `protected_roots`, and each selected source's `output_shape` and `crop_transform` from its pinned report. Keep the frozen proposal as historical evidence. The final reviewed plan and approval SHA must originate from the independent review, not from the files being checked for execution.

## Resource limits and recovery

| Resource | Pilot limits |
| --- | --- |
| Windows Job | One process, at most 2,560 MiB committed; 2 GiB peak working-set stop; at least 6 GiB host memory available at start |
| Time | At most 120 s per source transaction and 600 s for the supervised attempt, from child launch through preflight, commits and publication. Job close/kill and bounded reaping follow a stop. |
| Output | At most 4 MiB staging per source and 16 MiB logical output. Maximum 24 files including transaction bookkeeping. At least 256 MiB free throughout, plus logical write headroom at start. |
| Observed allocation | At most 64 MiB of measured regular-file allocation on a Windows local fixed drive. Measurement surrounds staged writes and publication. Infrastructure changes and a filesystem quota are unnecessary for this pilot. |

The working-set check samples the Windows peak counter. A source watchdog can terminate the child and leave the claim for reconciliation. Output allocation is sampled at transaction boundaries and excludes filesystem metadata and transient OS effects. It offers neither a hard quota nor a power-loss guarantee. Local-only paths, no network fetches, and offline worker settings cover the normal CLI; the Python guard is not a host firewall or an arbitrary-code sandbox.

Every source is exclusively staged, read back and pinned before publication. Complete sources can be reverified and skipped using the same plan, approval and output, with `--resume`. An orphan claim, unverified partial, unexpected path, invalid digest, or interrupted manifest link/unlink pair requires manual reconciliation. The CLI preserves these files and does not automatically retry. After a forced stop inspect retained files before relying on any outputs. Reader opens a complete generation only against an independently retained manifest SHA and checks product digests, crop/grid and mask consistency. It uses the existing research-only training interface. Formal R1 remains HOLD, and D6-B is deferred.

## Future Windows invocation

After independent review of this implementation and private plan, with separate authorization for real output, use the existing venv and explicit plan/approval SHA pins. The following placeholders are intentionally non-executable.

```powershell
.\.venv\Scripts\python.exe -B scripts/d6a3_produce.py `
  --plan "<reviewed private plan>" --plan-sha256 "<independently retained plan SHA>" `
  --root "<bound cache>" --evidence "<bound STAC directory>" --output "<bound fresh external output>" `
  --real-approval "<separately authorized token>" --real-approval-sha256 "<independently retained token SHA>"
```

Capture the printed manifest SHA and resource readings. Validate the output with `D6Data(output, manifest_sha256)` before any downstream use. The real producer has not been run on OAM pixels. End-to-end evidence to date uses four synthetic 2048×2048 inputs and the actual Windows Job and allocation APIs, with published products reopened by the D6 reader.
