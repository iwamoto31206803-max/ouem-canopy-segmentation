# D6-B segmented SatlasPretrain runbook

[Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100) defines the milestone. This PR prepares code and local inputs. Actual cloud transfers and training require the owner's separate approval of the run plan, operating limits and provider.

## Proposed frozen experiment

| Item | Proposal |
| --- | --- |
| Dataset | D6-A5 generation `9571151c0037d2ba7b05c9ad46a860579efa547d62a7248eccac7c96b47faf60`, 175 Train groups and 25 fresh Val groups, one fixed 224px patch each |
| Initialization | Official Aerial_SwinB_SI with pinned SHA, Swin-B + pretrained FPN + fresh official two-class SEGMENT head |
| Updates | Four seeded source-balanced cycles of 175, 700 total; each source once per cycle |
| Training | Batch 1, FP32, TF32/AMP off, AdamW 1e-4, weight decay 0.01, seed 0, all model parameters trainable, no scheduler/scaler/augmentation |
| Checkpoints | At updates 175/350/525/700 and unfinished segment boundaries with progress |
| Evaluation | Step 0 and 175/350/525/700 on separate fixed fresh Val25 and historical Val18; final 43 five-panel PNGs, intermediate metric JSON |
| Sessions | Start with up to 175 updates (provisional 1,800 s process budget, 300 s reserve and 12 GiB peak CUDA reserved ceiling on a suitable free runtime). Use 175 or 350 later where measured capacity permits. Plan at least one intentional cold restart before completion. |

Fresh official initialization gives an interpretable baseline and avoids inheriting the v5 update history. Continuing Step280 would answer a different fine-tuning question and would require selecting and disclosing different optimizer/lineage semantics. Keep Step280 intact. An optional separately authorized fixed Val25 read-only evaluation of Step280 can add context later.

Use the existing D5 loss, predictions, per-source TP/FP/FN/TN, IoU, F1, CE, predicted/GT fractions and defined-only macro and pooled micro metrics. Report final-minus-Step0 separately on each cohort. Any Step280 D5 comparison on Val18 draws on its historical recomputed CPU result and uses its documented device/provenance limits. Dataset and update count both change, so the experiment does not isolate a data effect. No performance gate or Val-driven tuning. Scope is OAM-TCD research, with Japan/GSI and commercial suitability unverified.

## Prepare locally and retain expected pins

Keep the source product generations, v5 and checkpoint evidence read-only. From a clean checkout of the reviewed executed commit, use a new private output directory outside Git. The public CLI example uses local path placeholders.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -B scripts/d6b_bundle.py pack --a5 "<A5 aggregate manifest>" --v5 "<v5 freeze root>" --archive "<fresh private archive.zip>" --commit "<full reviewed commit>"
& $python -B scripts/d6b_bundle.py unpack --archive "<archive.zip>" --sha256 "<retained archive SHA>" --output "<empty verification directory>"
& $python -B scripts/d6b_train.py --bundle "<verification directory>" --bundle-sha256 "<retained bundle.json SHA>" plan --run-id d6-a5-main700-01 --total 700 --save-every 175 --eval-every 175 --output "<fresh private run-plan.json>"
```

The archive contains verbatim A5, A4, pilot and shard manifests/rosters/receipts and their three TIFFs per patch, plus only the 54 fixed Val18 TIFFs and original v5 evidence JSON. A relocations table binds original reference path, digest and kind to a scratch path; no original manifest is edited. The unpacker checks archive and product SHA, exact inventory, and both cohorts. The run plan pins commit, the unpacked control SHA, initializer, dataset, scientific settings and the schedule hash. Keep the printed archive, control, plan and schedule pins separately. Keep the initializer separate and reuse an already verified authorized copy. The ZIP includes private manifests and must stay in the existing private hierarchy.

## Run, remote-read back and cold-restore

Use [the copy/paste Notebook](../../notebooks/OUEM_D6_Segmented_Satlas_Training.ipynb) after authorization. The Notebook assumes a prepared private Git checkout at the reviewed commit, as for D5; keep Git credentials and nontracked files out of staging. Select a free GPU manually, verify packages and the authorized private Drive location, and stage one ZIP and initializer to scratch. Make the run relay parent before the first segment. The trainer requires one GPU with SatlasPretrain 0.3.1 and retains the runtime's CUDA Torch. Record exact software versions, GPU and VRAM. A changed software receipt stops cold restore; GPU model changes with the same software can be observed, without assuming cross-device numerical identity.

1. **Session A.** Use a fresh scratch and unique session name. Run the first segment of the frozen experiment. Every committed point packages a full checkpoint with immutable run binding, complete schedule and next item, model, optimizer, Python/NumPy/Torch CPU/CUDA RNG, and fingerprint receipts. Save completes only at a completed update boundary. The segment limit and save reserve are checked between operations; they are soft wall-time bounds and cannot preempt an update, evaluation, upload or runtime eviction. VRAM is checked at phase boundaries. Recovery is from the last confirmed point, with any later updates reported as lost.
2. **Verify before release.** Step 0 evidence is published immediately. At each checkpoint, update logs and due evaluation are sealed with the checkpoint. Segment summaries have their own receipts. Relay writes publish the receipt last after a mount-path SHA check. Independently obtain bytes by Drive API file ID into fresh scratch and validate against the pins printed by the local trainer. Mount-cache reads alone do not establish remote durability. If upload or verification fails, keep the prior confirmed checkpoint and retain the local error/partial results for inspection. Save the per-session timing, resource, software and remote transfer receipts.
3. **Session B and onward.** After a confirmed checkpoint, record the session/runtime identity and actually disconnect/delete the runtime so session A scratch is gone. Start a new allowed runtime and new session name. Restage inputs and fetch the selected prior checkpoint directory from Drive into fresh scratch. Use its independently retained receipt SHA. The strict restore verifies lineage, next item, schedule, parameters, optimizer and RNG before continuing. New outputs use fresh local and relay names. Follow one verified lineage and do not launch concurrent continuations. Keep the last-known-good and preceding confirmed point until archived.
4. **GEEKOM-A6 archive.** Obtain the private remote-readback directories through the established authorized route and verify/copy each from local downloads as shown below. Include Step 0, all retained boundary evidence, final checkpoint, segment outputs and final comparison. Keep the plan, bundle and schedule pins and record the runtime IDs and proof of cold reset. Compare checksums on the durable archive before any owner-controlled Drive cleanup. No pruning is automatic.

```powershell
.\.venv\Scripts\python.exe -B scripts/d6b_relay.py "<downloaded record directory>" --receipt-sha256 "<independently retained receipt SHA>" --archive "<fresh GEEKOM archive directory>"
```

## Resource budget and acceptance

A D5 checkpoint measured 1,066,503,375 bytes. Two retained recovery points plus a new upload would require about 3.2 GB for checkpoint files alone. Include the separate initializer, data archive, scratch working and remote-readback copies, and evidence. The Notebook proposes a 4 GB Drive free allowance and 6 GiB scratch floor as initial planning values. Check actual Drive quota, available runtime, VRAM and scratch before execution; measure new checkpoint bytes, stage/upload/remote-readback throughput, updates/second, total session time and any lost work. Free-tier allocation and capacity can change. No paid services or quota workarounds are included.

Milestone PASS requires one frozen run completing 700 updates across at least two genuinely distinct GPU sessions, verified remote bytes and a cold restore that continues from the recorded schedule position. Confirm no duplicate/skipped committed updates and archive the final checkpoint, separate validation results and logs with verification on Drive and GEEKOM. Comparing environments and numerical portability is part of the report; GPU equal-bit continuity is unverified. Local CPU surrogate results alone cannot satisfy this milestone. Formal R1 remains HOLD and GT54 stays deferred/unverified.

## Local preparation record

On GEEKOM-A6 the staged proposal archive was **31,331,482 bytes** with 1,078 payload files and 32,980,057 bytes after extraction including its control file. The separately cached initializer was **504,439,674 bytes**, with its official SHA verified. Archive build took 7.83 s and extraction with CPU Reader/schedule checks took 28.62 s. The relocated Reader supplied all 175 Train, 25 fresh Val and 18 separate historical Val patches, and the proposed 700-update schedule visited each Train group once per cycle. These measurements cover local storage and CPU readback, while Notebook staging, remote durability and GPU resources remain for the authorized run.

| Local proposal pin | Value |
| --- | --- |
| Executable Git revision | `9ad5d74c6cc5dfc0c6c7fd3354dba4ef13e9af45` |
| ZIP SHA-256 | `0066fe8f800e43c7625cdd42498935ed8fdbd2e687079247329c9118ec57b5db` |
| Extracted bundle control SHA-256 | `b2229155ceeb24be4b798b11c0dbe3c0b2133fd79775429af48adff436165667` |
| Proposed run plan SHA-256 | `37818de053c5668e19feddb9196ae8d5233720bf2b9c9272d7d8ea2152bfa168` |
| Schedule SHA-256 | `916e46817e0f24d3007711f81f8c0ea1d9f44a5999b57751d42dcbb9607ff532` |

If the approved code pin or experiment settings change during review or squash merge, build a new local ZIP/control/plan receipt for the approved revision before the first run. Preserve the staged proposal and never change these pins within an active lineage.

Windows validation passed **791 tests**, with two existing symlink-privilege skips and zero failures. The focused D6-B/A5/D5 tests passed 18. CPU tests covered split versus uninterrupted state fingerprints, reloading from a different path after deleting the old scratch, corruption/config/session mismatch, time reserve, interruption with last-good preservation, and A5 path relocation. The Notebook's seven code cells passed static compilation, with no executed cells. The full suite's one visible warning was the expected duplicate-ZIP fixture; pending deprecation warnings were filtered.
