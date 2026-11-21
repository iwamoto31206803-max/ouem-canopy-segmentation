# D6 planning baseline — deterministic preprocessing and restartable GPU operations

**Date:** 2026-11-21 (JST)  
**State:** **PLANNING BASELINE (reviewed decision update 2026-11-21)**. **D6-A preprocessing is the next implementation/design focus; D6-B implementation is deferred to the next separately approved training experiment, after D6-A.** D6-B tasks are captured in [Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100). This document does not approve new data acquisition/processing, GPU training, expenditure, model/label changes or Formal R1 promotion.  
**Authority:** [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md), [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md); for actual execution status use [Current State](CURRENT_STATE.md). D6-A/B are **research workstream names**, not changes to the Charter's D0–D5 acceptance gates.

## 1. Motivation and immutable starting point

The D5 exploratory research run `d5-v5-main280-01` completed **280 training updates** using SatlasPretrain `Aerial_SwinB_SI` SwinB + FPN + fresh two-class `Head.SEGMENT`. The Step280 model was recovered on the GEEKOM-A6 CPU, producing spatially meaningful canopy predictions on some source-domain images, with substantial image-to-image variation. It established **initial learning feasibility and a working checkpoint/evaluation/evidence pathway**, not practical accuracy or improvement over an independently controlled earlier experiment.

- Frozen exploratory research v5: **Train 70 source images / 139 windows; Val 18 sources / 18 fixed windows; 88 sources / 157 windows / 471 GeoTIFFs**. It must remain unchanged for historical comparison.
- Step280 fixed Val18 recomputation: **18 × 224 × 224 = 903,168 pixels**; micro tree IoU **0.6156592718468059**, macro IoU **0.40682443455533224** (defined on 16 windows), micro F1 **0.7621152337931579**. These are reference observations, not acceptance thresholds or national accuracy estimates.
- Complete local evidence and independent **22-file Drive remote readback** were confirmed. See [D5-B2 execution and archival record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md) and [Issue #95](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95).
- Historical trainer/evaluator, v5 inputs, original checkpoint and private evidence remain immutable. **Formal R1: 0/150 accepted; HOLD unchanged**. The 18 research validation windows are not an independent positive-tree Japan/GSI evaluation.

**Owner-approved research direction:** pause detailed fixed-Val18 FP/FN root-cause and threshold analysis; prioritize planning for more data, stronger preprocessing reproducibility and feasible larger GPU training. Do **not** infer that increasing data or updates must improve quality.

## 2. D6-A — objective, scalable preprocessing and research admission

**Owner clarification (2026-11-21):** "reproducible across PCs" primarily means **the same predefined rules select/process the same class of eligible inputs without per-image human/Codex visual judgement**, not a rigorous byte-identical Windows/Linux TIFF reproducibility project. **Top priority: machine-actionable QA/admission and high-throughput, bounded preprocessing**. For the actual old-rule inventory, per-image visual-dependency risks, proposed decision statuses and first design scope, see [D6-A objective QA audit](implementation/D6A_OBJECTIVE_QA_POLICY_AUDIT.md). This is a *proposal*, not new automatic acceptance already implemented.

**Research label-trust policy (owner decision 2026-11-21):** use official published OAM-TCD annotation as-is as the working teacher for exploratory training. **Minimize automatic exclusions to explicit technical, source-policy, source-split or geometric/crop conditions**; do not ask a person or Codex to visually certify each normal image. Label noise is expected and should be assessed in aggregate/with optional sampled audits, not endlessly corrected image-by-image. License/attribution declarations must still be checked mechanically, and research eligibility remains distinct from Formal R1 or commercial release. [D6-A1 implementation backlog](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/103).

**Latest implementation status (2026-11-21):** [D6-A1 PR #105](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/105) is **merged**, adding [`scripts/d6a_research_qa.py`](../scripts/d6a_research_qa.py), [runbook](implementation/D6A1_RESEARCH_AUTO_QA.md), and synthetic tests. It only generates source/window eligibility decisions and crop recipes; **no real OAM-TCD QA pilot, new training products, or GPU training has been run**. The next stage is the **unexecuted, separately gated** [D6-A2 pilot, Issue #106](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/106). The earlier inventory below remains useful as the design history.

### Existing foundation

- `scripts/research_patches.py`: deterministic per-ID/seed crop positions, geometric black-support exclusions, area resampling, fraction/binary mask, product readback/hashes. It currently creates `coverage_review=PENDING_VISUAL_REVIEW`.
- `scripts/research_data.py`: immutable input/crop/review identifiers, grid and SHA-256 checks, source-balanced reader. **Its legacy admission requires explicit `CLEAR_FOR_EXPLORATION` reviewer decisions**; preserve the historical contract, build a **separate versioned D6 research pathway** for objective automatic eligibility.
- `scripts/m1_5d_zenodo_prepare.py`, `scripts/m1_5b_pilot.py`, `scripts/r1b_qa_runner.py` and `scripts/m1_5b2_border_audit.py`: already calculate machine-based rights/metadata consistency, source format/grid/validity, source-group independence and exact-black components. **They do not certify all photographed coverage, complete tree truth or Formal R1 admission.**
- `scripts/r1_dataset.py` is the **Formal R1** contract with separate source-bound review/acceptance assertions; **do not rewrite or automatically promote R1 HOLD**.

### Primary requirements

1. **Predeclare a versioned research-only policy:** source generation, license/attribution metadata requirements, permitted official folds, source-group independence, upstream Tree+Canopy union semantics and known category-crosswalk limitations, RGB/mask/CRS/grid rules, crop/black-support policy, seed, target GSD and thresholds. The policy is approved before batch selection/training, **not tuned from model predictions**.
2. **Replace per-image visual-gated CLEAR/HOLD selection with objective, reasoned decisions:** `AUTO_ELIGIBLE_RESEARCH`, `AUTO_HOLD`, `AUTO_EXCLUDE`/`TECHNICAL_FAIL` plus orthogonal `UNASSESSED_SEMANTIC_RISK`. For each status log a rule ID, measured value, expected value/threshold, input/algorithm/policy identity and limitations. Do not allow arbitrary `looks good`, `AI visual confidence` or silent user/Codex overrides. A machine-pass is **not** a claim of semantically correct tree truth or complete photographic coverage.
3. **Treat ambiguous coverage honestly:** use published valid-data/footprint evidence when available; otherwise use prespecified numeric/geometric rules to **exclude candidate windows intersecting hypothesized border-connected black support**, or return a rule-based HOLD. The historical >=1% border-black flag is a heuristic, **not guaranteed nodata detection**; white roofs, internal shadows, zero/all-one teachers are not automatically rejected solely for their colors/label prevalence. Don't auto-edit labels or fabricate valid fractions.
4. **Automate source-level batches and durable receipts:** bounded I/O/memory/time, stable source/crop keys independent of processing order, per-source PASS/HOLD/FAIL reasons, integrity checks, idempotent restart from verified products, immutable dataset/manifest identities and source-wise counts. Control repeated/overlapping crops and source diversity; many patches are not many independent source images.
5. **Keep people at policy/audit level, not per-image gate level:** owners review the policy and dataset/source rights scope; independent **stratified sample audits** examine teacher noise and how frequently automated checks fail. Material systematic problems lead to a **new versioned policy** and rerun, not undocumented per-image exceptions. Formal R1 review remains separate.
6. **Keep routine computational reproducibility:** fixed seeds/code/config, retained output hashes, array/grid integrity and package versions. An optional small cross-PC smoke check can validate ordinary portability; **neither TIFF byte identity nor exact floating-point bit equality across OS/BLAS is a primary D6-A entry gate**.

### Work sequence and evidence

- **A0, now — inventory/policy review:** [D6-A audit](implementation/D6A_OBJECTIVE_QA_POLICY_AUDIT.md) lists existing automatic vs visual conditions, missing semantic guarantees and proposed objective research-only statuses. Review policy risks; don't call them accepted before approval.
- **A1, IMPLEMENTED/REVIEWED in PR #105 — objective decision engine:** synthetic fixtures cover edge/internal black, exact-white features, official holdout, duplicate/near-source, missing STAC, grayscale/invalid TIFF, source hash/CRS failures, no eligible crop, all-zero/all-one masks and reordered processing. No real mass data processing.
- **A2, separately approved bounded real-data pilot:** evaluate automated eligibility counts, HOLD/EXCLUDE reason distribution, source diversity, overlap, peak RAM, throughput and restart/idempotence. Compare with old Phase C human/Codex assessments as an **independent diagnostic**, not manual approval overrides.
- **A3, after review — expand mechanically eligible dataset** with stable source-level selection and frozen Train/Val separation; preserve original research v5/Val18 and Formal R1 artifacts unchanged. Coordinate final artifact format and transfer with deferred D6-B.

**Proposed D6-A gate (not yet PASS):** `OBJECTIVE_RESEARCH_ADMISSION_POLICY_REVIEWED` + `NO_PER_IMAGE_VISUAL_GATE_TESTED` + `RULE_REASON_AND_FAIL_CLOSED_TESTS_PASS` + `BATCH_YIELD_AND_IDEMPOTENCE_MEASURED`. No acceptance claim is made by this document.

## 3. D6-B — free-tier-aware, durable and resumable GPU workflow

**Objective (deferred implementation):** run larger training while treating a free Notebook GPU and its local filesystem as **ephemeral**, without making free GPU-hours, a fixed GPU model, a persistent session or free cloud capacity an assumption. **Owner's preferred policy:** exhaust practical, permitted free GPU sessions first, pausing at verified checkpoints and resuming whenever quota permits; evaluate a paid GPU only if measured waiting/overhead is excessive, under a new owner-approved budget. **Google Drive is the intended relay/short-term recovery store (not Google Cloud Storage); GEEKOM-A6/external storage is the long-term archive.** No D6-B code is needed until the next training campaign is designed. See [deferred implementation checklist, Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100).

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

**Order selected (2026-11-21): D6-A first.** Begin with objective rule-based automatic eligibility and scalable preprocessing without image-by-image visual gates; an ordinary cross-PC smoke check is secondary. **Implement D6-B only as part of preparation for the next approved training experiment**, preferably with the smallest cold-resume/Drive-transfer functions needed. D6-B's checklist is preserved in Issue #100, rather than initiating a separate implementation campaign now. The data-processing scale decision depends on reviewed D6-A evidence; substantive training still depends on verified D6-B operational safety and an explicitly frozen dataset. A small GPU pilot may inform hardware sizing only after separate approval.

| Decision | Current disposition |
| --- | --- |
| D5 Step280 initial-learning feasibility | Achieved in the bounded exploratory sense; not practical accuracy |
| D5 Step280 full Val18 evidence and remote preservation | **PASS**, frozen [archival record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md) |
| D6-A objective auto-QA rule and scalable preprocessing | **A1 code merged, A2 real-data pilot NOT STARTED**; [PR #105](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/105), [A2 Issue #106](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/106); still no training product generation or real-data conversion |
| D6-B interruption/resume + resource budget trial | **DEFERRED: implement/test when next approved training experiment is prepared** ([Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100)) |
| New dataset/source rights/label acceptance | **PENDING**, no automatic promotion |
| Enlarged GPU training campaign | **NOT AUTHORIZED** |
| Japanese positive-tree independent validation | **PENDING** |
| Formal R1 | **HOLD**; keep prior 150-image candidate decisions unchanged |

Engineering gates (execute/load/restore/evidence), research gates (controlled held-out accuracy/diversity, including Japan) and Formal R1/commercial gates are **separate**. Report all three rather than collapsing a successful train run into production readiness. Do not choose model settings based on repeated fixed-Val18 probing.

## 5. Planning deliverables before implementation

- **D6-A policy review:** [rule-based objective QA audit](implementation/D6A_OBJECTIVE_QA_POLICY_AUDIT.md), research-only eligible/HOLD/EXCLUDE reason schema, decisions independent of per-image visual judgements, separate semantic-risk flags, synthetic cases, resource/IO budgets, idempotent resume and dataset-yield measurements. Cross-PC TIFF byte identity is not a required gate.
- **D6-B deferred training-coupled review:** refer to [Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100) at the next training design stage. Prefer free Notebook GPU + Google Drive short-term relay + GEEKOM long-term archival; measure limits, staging/shard capacity, checkpoint-retention/remote verification, cold-resume test and a potential *optional* paid-compute ceiling before owner authorization.
- **Joint decision:** select data volume and diversity target *after* sources/rights/labels and resource budget are known; establish independent Japan-positive evaluation set; keep Step280/v5 outputs an immutable baseline.
- **Review workflow:** separate small implementation PRs and explicit owner execution gate; no automatic data download, remote upload, GPU use, model retraining or Formal R1 promotion from this document.

**Useful references:** [README](../README.md); [Current State](CURRENT_STATE.md); [Next Roadmap](ROADMAP_NEXT.md); [D5-B2 record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md); [Colab FAQ](https://research.google.com/colaboratory/faq.html); [Google storage help](https://support.google.com/googleone/answer/9004014). Provider conditions must be checked fresh before operational use.
