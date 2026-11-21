# D6 planning baseline — deterministic preprocessing and restartable GPU operations

**Date:** 2026-11-21 (JST)  
**State:** **DRAFT FOR REVIEW / PLANNING ONLY**; **D6-A versus D6-B order is not decided**. This document is not approval to acquire more data, train, spend money, modify labels, use a new external service, or promote Formal R1.  
**Authority:** [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md), [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md); for actual execution status use [Current State](CURRENT_STATE.md). D6-A/B are **research workstream names**, not changes to the Charter's D0–D5 acceptance gates.

## 1. Motivation and immutable starting point

The D5 exploratory research run `d5-v5-main280-01` completed **280 training updates** using SatlasPretrain `Aerial_SwinB_SI` SwinB + FPN + fresh two-class `Head.SEGMENT`. The Step280 model was recovered on the GEEKOM-A6 CPU, producing spatially meaningful canopy predictions on some source-domain images, with substantial image-to-image variation. It established **initial learning feasibility and a working checkpoint/evaluation/evidence pathway**, not practical accuracy or improvement over an independently controlled earlier experiment.

- Frozen exploratory research v5: **Train 70 source images / 139 windows; Val 18 sources / 18 fixed windows; 88 sources / 157 windows / 471 GeoTIFFs**. It must remain unchanged for historical comparison.
- Step280 fixed Val18 recomputation: **18 × 224 × 224 = 903,168 pixels**; micro tree IoU **0.6156592718468059**, macro IoU **0.40682443455533224** (defined on 16 windows), micro F1 **0.7621152337931579**. These are reference observations, not acceptance thresholds or national accuracy estimates.
- Complete local evidence and independent **22-file Drive remote readback** were confirmed. See [D5-B2 execution and archival record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md) and [Issue #95](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95).
- Historical trainer/evaluator, v5 inputs, original checkpoint and private evidence remain immutable. **Formal R1: 0/150 accepted; HOLD unchanged**. The 18 research validation windows are not an independent positive-tree Japan/GSI evaluation.

**Owner-approved research direction:** pause detailed fixed-Val18 FP/FN root-cause and threshold analysis; prioritize planning for more data, stronger preprocessing reproducibility and feasible larger GPU training. Do **not** infer that increasing data or updates must improve quality.

## 2. D6-A — machine-independent, scalable data preprocessing

**Objective:** given the **same pinned source bytes + same reviewed recipe**, processing on two authorized PCs produces **semantically identical RGB, canopy-fraction/mask pixels and spatial metadata**, without an operator manually choosing crops or intervening per image. File-byte identity is a desirable secondary objective, not silently assumed.

### Existing reusable foundation

- `scripts/research_patches.py`: explicit source grid, integer support checks, `random.Random(f'{seed}:{image_id}')` deterministic window choice, area downsampling, fractional canopy and binary threshold, TIFF readback, per-product hashes and reviewed plan output. Existing source choices are still bounded and require accepted/reviewed inputs.
- `scripts/research_data.py`: source/crop/review/rights checks, frozen immutable hashes and source/fold consistency; rehashes and validates products on use. `scripts/r1_dataset.py` provides a separate **Formal R1** contract—research CLEAR does not equal formal PASS.
- `scripts/r1b3_batch.py` and QA modules already show bounded offline batches and ledger checks. **Reuse principles, not unreviewed promotion or a parallel unchecked pipeline.**

### Design requirements (to validate, not yet implemented)

1. **Pin the full recipe and input contract:** source revision/row ID/`oam_id`, license/attribution, source RGB and annotation SHA-256, category mapping (Individual tree + Tree canopy; void/background semantics), spatial reference, target GSD/grid alignment, resampler, RGB rounding, fractional aggregation, mask threshold, crop policy, seed and split/group identity.
2. **Enforce source-level independence:** original `oam_id` and overlapping/near-duplicate scenes must not leak between Train and Val/test. Keep official OAM-TCD holdout and GT54 out of model/threshold tuning. Preserve original fixed Val18 as an **unchanged historical comparison set**; plan a truly independent, quality-reviewed validation set including **Japanese positive-tree scenes** before a claim about Japan.
3. **Bound memory and labor:** chunk/batch source imagery without loading the whole pool, report source-level PASS/HOLD/FAIL and reasons, generate bounded preview contact sheets for **targeted human review**, and never auto-accept ambiguous borders, nursery/planting rows, grayscale TIFFs, labels or rights.
4. **Make selection independent of execution order:** fixed per-source/per-crop keys and deterministic stable sorting; avoid shared/global RNG, time/random file naming or filesystem iteration order for data identity. Resume/idempotent skip decisions must verify existing products against a declared manifest, not silently overwrite.
5. **Lock/record the environment:** Python, NumPy, Pillow, Rasterio/GDAL/PROJ, compression driver/options, CPU feature/OS where relevant, algorithm code commit and any container/image digest. Do not claim hardware independence before exercising two actually different environments.
6. **Write immutable output in two stages:** fresh staging area; product readback; canonical manifest, content hashes, summary and signed-off QA; only then distinct versioned **dataset freeze**. No in-place mutation of v5. Stop on discrepancies rather than best-effort correction of teacher labels.
7. **Use two reproducibility tiers:**
   - **Required (content):** pixel array values, shapes, dtype, mask semantics, CRS, affine, bounds, target GSD, selected `crop_xywh`, split assignments and ordered canonical manifests match on both PCs. Compute independent array-content SHA-256 on a clearly documented canonical serialization when TIFF container bytes differ.
   - **Desired (container):** byte-identical GeoTIFF and artifact SHA-256. GDAL/TIFF metadata, library versions and DEFLATE implementation can change container bytes; report a difference explicitly, never call it a content mismatch or quietly waive it. If contractual byte identity becomes necessary, use a locked containerized writer and test it.

### D6-A acceptance experiment

Take a small **rights-reviewed, diverse fixed subset** of sources (e.g., dense and sparse crowns, borders, all-background, distinct source resolutions, grayscale HOLD as an expected reject). Run the **same immutable source bundle and exact recipe on GEEKOM-A6 and a second authorized PC/runtime**; compare canonical array and spatial metadata hashes, window choices, manifest contents, reproducibility under shuffled processing order, and expected fail-closed cases. Record versions, resource usage, discrepancies, byte-vs-content verdict, and independent readback. **No large production run until the differential test is reviewed.**

**Gate D6-A:** `PREPROCESS_CONTENT_REPRODUCIBLE_ON_TWO_ENVIRONMENTS` and `FAIL_CLOSED_QA_VERIFIED` (names proposed; not yet PASS), with any non-identical GeoTIFF bytes explicitly documented. D6-A is not a license or Formal R1 gate.

## 3. D6-B — free-tier-aware, durable and resumable GPU workflow

**Objective:** run larger training while treating a free Notebook GPU and its local filesystem as **ephemeral**, without making free GPU-hours, a fixed GPU model, a persistent session or free cloud capacity an assumption.

### Separation of responsibilities

| Layer | Role and durability contract |
| --- | --- |
| GEEKOM-A6 / authorized CPU hosts | Maintain licensed/pinned source cache, prepare and verify data, review QA; retain a local backup outside Git and the temporary GPU runtime. |
| Transfer archives / immutable manifest | Publish versioned bounded shards with source/product/manifest SHA-256, exact code/config identity and ownership/attribution; record staged location and sizes. |
| GPU Notebook scratch | Download **only needed approved shards**, hash-check and unpack locally, preflight CUDA/package versions, and train with finite resource and step limits. Scratch/Notebook state may vanish at any time. |
| Durable storage | Save checkpoints, complete training state, provenance, evaluation and receipts outside the session; verify an **independent readback** before marking a recovery point safe. Google Drive and existing local copy are possible layers, subject to available quota and privacy/access checks. |

**Checks before selecting a provider (Colab, Kaggle or later paid alternatives):** current terms and availability, session interruption behavior, GPU model/VRAM, Python/CUDA/Torch/Satlas compatibility, aggregate upload/download quotas, free storage remaining, data/weight license and acceptable security posture. **The service-specific free quota may change and must be measured or checked at the time of each approved run.** No fixed free-hour or guaranteed 12-hour session is assumed.

### Sizing and recovery

1. **Capacity ledger before upload:** measured source archive bytes, expanded training scratch, number/size of sharded packages, checkpoint size × retained count, temporary copy/atomic-write overhead, logs/PNG/evaluation, and free Drive space shared with other Google services. Report expected peak, not only final total. A Step280 checkpoint was **1,066,503,375 bytes**, a historical reference—not a hard future size.
2. **Budget bounded computation:** measured preflight step throughput/VRAM and loading time, controlled batch/gradient accumulation and checkpoint cadence; choose a conservative run segment shorter than an unreliable session horizon, without assuming linear scaling across hardware.
3. **Fully restorable checkpoint:** strict model state, optimizer/scheduler/scaler as used, completed update, sampler/dataset position and immutable schedule, Python/NumPy/CPU/CUDA RNG, code/dataset/initializer/config identity and environment/precision flags. Handle dataloader ordering deterministically or record any inability to provide exact continuation.
4. **Failure-safe checkpoint publication:** write to a new temporary/scope-limited output, validate finite state and SHA-256, upload to durable destination and independently read/download/hash it; mark the last confirmed checkpoint only after verified readback. Do not overwrite a known-good checkpoint, and never mistake local scratch/Drive sync success for remote durability.
5. **Interruption drill:** intentionally stop a **small, separately authorized** CUDA run, discard scratch/session, restore from the last independently verified remote checkpoint on a fresh runtime and compare required state/sampler continuity and permissible numerical differences. A differing environment is an explicit compatibility evaluation, not automatic proof of bitwise equality.
6. **Fallback policy:** if free tier cannot hold input/shards/checkpoints or measured training exceeds reliable run windows, report a costed alternative (different shard strategy, retention policy, provider or paid GPU). **No payment/provider switch without owner approval.**

**Gate D6-B:** `RESOURCE_BUDGET_MEASURED`, `CHECKPOINT_REMOTE_READBACK_PASS` and `INTERRUPTED_RESUME_DEMONSTRATED` on a small controlled GPU pilot (proposed; not yet PASS). Training scale and any full GPU campaign require subsequent separate owner approval.

## 4. Execution decision, evidence hierarchy and stop rules

**No order has been selected between D6-A and D6-B.** The two can be designed independently. A data-processing scale decision depends on reviewed D6-A evidence; a substantive training run depends on verified D6-B operational safety and an explicitly frozen dataset. A small run may inform hardware sizing only after separate approval.

| Decision | Current disposition |
| --- | --- |
| D5 Step280 initial-learning feasibility | Achieved in the bounded exploratory sense; not practical accuracy |
| D5 Step280 full Val18 evidence and remote preservation | **PASS**, frozen [archival record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md) |
| D6-A two-runtime preprocessing equivalence | **NOT STARTED** |
| D6-B interruption/resume + resource budget trial | **NOT STARTED** |
| New dataset/source rights/label acceptance | **PENDING**, no automatic promotion |
| Enlarged GPU training campaign | **NOT AUTHORIZED** |
| Japanese positive-tree independent validation | **PENDING** |
| Formal R1 | **HOLD**; keep prior 150-image candidate decisions unchanged |

Engineering gates (execute/load/restore/evidence), research gates (controlled held-out accuracy/diversity, including Japan) and Formal R1/commercial gates are **separate**. Report all three rather than collapsing a successful train run into production readiness. Do not choose model settings based on repeated fixed-Val18 probing.

## 5. Planning deliverables before implementation

- **D6-A design review:** deterministic crop/label/grid specification, dependency lock, cross-machine differential test fixture and expected PASS/HOLD/FAIL cases, resource/IO bounds and exact canonical-hash contract.
- **D6-B design review:** candidate provider and dynamic quota check, staging/shard plan, peak storage/VRAM/throughput measurements, checkpoint-retention and remote verification, cold-resume failure-injection protocol and cost ceiling.
- **Joint decision:** select data volume and diversity target *after* sources/rights/labels and resource budget are known; establish independent Japan-positive evaluation set; keep Step280/v5 outputs an immutable baseline.
- **Review workflow:** separate small implementation PRs and explicit owner execution gate; no automatic data download, remote upload, GPU use, model retraining or Formal R1 promotion from this document.

**Useful references:** [README](../README.md); [Current State](CURRENT_STATE.md); [Next Roadmap](ROADMAP_NEXT.md); [D5-B2 record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md); [Colab FAQ](https://research.google.com/colaboratory/faq.html); [Google storage help](https://support.google.com/googleone/answer/9004014). Provider conditions must be checked fresh before operational use.
