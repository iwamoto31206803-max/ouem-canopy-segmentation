# OUEM Canopy Segmentation — Next Roadmap

Revision **2026-11-26** — D5 and D6-A5 preserved; D6-B implementation ready for review. This roadmap remains subordinate to the [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md) and [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md). The Charter's M1/M2/D0–D5 scope remains in force; D6-A/B here name *research workstreams*, not new formal acceptance gates.

## Current work order — review D6-B preparation for segmented training (2026-11-26)

**Achieved in the bounded exploratory sense:** v5 Train **70 originals / 139 windows**, Val **18 originals / 18 fixed windows**; original D5 **280-update GPU run** completed; original Step280 checkpoint pinned and recovered on GEEKOM-A6 CPU. Visually meaningful source-domain canopy predictions were generated. Full fixed Val18 recomputation covered **18 windows / 903,168 pixels**, reporting micro IoU **0.616** and macro IoU **0.407**; **22 evidence files** passed local-copy and independent private Drive remote readback checks. [D5-B2 record](implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md), [Issue #95](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95).

**Boundary:** this proves preliminary learning feasibility and the recovery/evidence process, **not** representative accuracy, original GPU-to-CPU output equivalence or Japanese target-domain suitability. **Formal R1 remains 0/150 accepted / HOLD.** The owner paused detailed fixed-Val18 FP/FN causal, threshold and architecture analysis until a larger, better-controlled experiment can inform it. Preserve Step280 and frozen v5 as immutable historical references.

| Proposed workstream | Immediate planning deliverable | Gate before scaling |
| --- | --- | --- |
| **D6-A1 — research auto-QA (merged)** | [PR #105](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/105): offline machine-only source/window QA and deterministic crop recipes; labels-as-is, no image-level human/Codex gate | Code/synthetic reviewed; QA-only tool, not a training TIFF generator |
| **D6-A2 — finite real-data QA pilot (COMPLETE)** | [Issue #106](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/106) closed: 24 cached sources, Train 16/18 + Val 4/6 research eligible, four edge-black geometric HOLD; two runs match per-source statuses/reasons/crop coordinates, guarded RAM ~584 MiB and time <30 s per run (Codex reported) | **Bounded research-only QA evidence**; nonrepresentative filtered cohort, private receipts not independently reviewed, source authenticity mostly unverified |
| **D6-A3-0 — producer/reader design (COMPLETE)** | [Issue #108](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/108) reviewed/closed: K=1 A1-selected source crop, three GeoTIFFs, verified source publication, frozen manifest, full-grid-coordinate D6 reader, guarded resources | **Design reviewed**; no code or data production in A3-0 |
| **D6-A3-1 — synthetic-only producer/reader (COMPLETE)** | [Issue #110](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/110) closed; [PR #112](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/112) merged, K=1 224px synthetic fixture generator, SHA receipts, Windows Job, restart and D6 reader. Codex reported full 750 PASS/2 SKIP | No real imagery input support, real product generation, mass preprocessing or GPU enabled |
| **D6-A3-2 — four-source real pilot (PASS)** | Authorized Windows CPU pilot generated four 224×224 patches (2 new Train / 2 fresh Val); D6Data read back all 12 TIFFs. One external expected SHA and three local-only SHA. Fixed Val18 and official Test protected; GT54 deferred/unverified. | Review the [pilot record and next CPU batch proposal](implementation/D6A3_BOUNDED_PILOT.md) in [Issue #113](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/113). Bulk processing and GPU execution await separate decisions. |
| **D6-A4 — cached generation verified** | 80 distinct groups (75 Train, 5 fresh Val), 240 TIFFs and one CPU schedule cycle verified. | [D6-A4 result and Reader contract](implementation/D6A4_CACHED_GENERATION.md). [Issue #116](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/116). The follow-up is recorded in D6-A5 below. |
| **D6-A5 — 200-source research generation verified** | 175 Train / 25 fresh Val, 600 TIFFs, reusable private source inventory and bounded metadata refresh. | [Inventory rebuild and generation record](implementation/D6A5_OAM_INVENTORY.md). [Issue #118](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/118). GPU experiment preparation remains separate. |
| **D6-A — objective automated QA and scalable preprocessing** | Define a machine-evaluable **research-only AUTO_ELIGIBLE/HOLD/EXCLUDE policy** from source rights declarations, original-source grouping, raster/label encoding, coverage diagnostics and reproducible crop rules; automate bounded batches without per-image human/Codex visual approval. [Existing-rule audit](implementation/D6A_OBJECTIVE_QA_POLICY_AUDIT.md) | Rules and reason evidence reviewed, synthetic fail-closed tests, idempotent and bounded batch pilot with eligibility/yield statistics. Human role is policy/sample audit, not each image; exact cross-PC TIFF bytes are **not** the gate |
| **D6-B — segmented GPU milestone** | Proposed 700 updates on fixed A5, fresh Val25 and historical Val18 evaluated separately. | [D6-B segmented training runbook](implementation/D6B_SEGMENTED_TRAINING.md). [Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100). Cloud transfer and at least two actual GPU sessions await separate approval. |
| **Joint data and experiment plan** | Determine target volume and diversity from measured availability and resource budget; preserve source-level Train/Val separation and official holdout; plan independent Japan-positive evaluation | Fresh approved dataset freeze, evaluation policy and owner decision before full GPU training; Formal R1 acceptance remains separate |

**Historical ordering decision (2026-11-21):** prioritize **D6-A objective auto-QA and the mechanical source-to-patch pipeline** first; routine result consistency and portability checks are secondary. **Defer D6-B implementation until preparing the next separately approved training experiment**, then integrate free Notebook GPU cold-resume, Google Drive as the short-term relay, and GEEKOM-A6 as the durable archive. The deferred B1–B6 implementation/recovery checklist lives in [Issue #100](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/100); no independent D6-B coding campaign is required now. Evaluate paid GPU only if measured free-tier delays warrant it, with separate approval. Changes in source, label, external service or GPU execution still require a reviewed gate. Preserve the SwinB + FPN + SEGMENT2 baseline; no assumption that more data or steps necessarily improves quality.

Use [D6-A/B detailed baseline](D6_PREPROCESSING_GPU_OPERATIONS_PLAN.md) and [D6-A Phase C/R1b rule audit](implementation/D6A_OBJECTIVE_QA_POLICY_AUDIT.md) for the objective eligibility/QA contract and later GPU/storage-resilience criteria. **D6-A1 code is in main and D6-A2 has completed a bounded real-image QA-only pilot**; A3-2b verified four research-only patch sets, A4 integrated them with 76 new patches, and A5 reached 200 groups. D6-B local implementation is ready for review. New GPU training remains pending.

## Historical roadmap — 2026-11-18 Phase C closure

The following Phase C execution plan and subsequent 2026-11-07/14 R0–R5 snapshot are preserved for decision provenance. References below to *next* GEEKOM preflight, new v5 CUDA bridge or a GPU run *pending* were accurate on 2026-11-18 but are **superseded by the D5 record above**.

## 2026-11-18 research-only path after Phase C

Phase C1–C8 data expansion and the Issue #81 scoped scientific review are complete. The current baseline is research v5, Train 70 originals / 139 windows and Val 18 originals / 18 fixed windows. Formal R1 remains 0/150 accepted, all HOLD. Bounded exploratory learning received a **CONDITIONAL GO recommendation**, with execution approval pending. [Current State](CURRENT_STATE.md), [closure review](PROJECT_REVIEW_20261118.md), [independent scientific disposition](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971582740)

| Step | Smallest next work | Decision gate |
| --- | --- | --- |
| 1. GEEKOM evidence and image confirmation | Locate external v5 and private scientific-review receipts, compare reviewed pins with measured hashes, and review the 24 CLEAR + 9 held contact-sheet windows. Prioritize 3380/4066 nursery crowns, 802/727 planting rows, 2006/614/4139 linear positives, 2991 and 4315. Record human decisions and limits while preserving v5/Val18. | Local evidence reconciled and targeted human image judgments recorded. A discrepancy becomes a scoped follow-up before execution. |
| 2. Explicit v5 experiment plan | Pin dataset/plan/review identity, official initializer and rights evidence, fixed Val18, source-balanced schedule, seed, finite steps/resources, CUDA runtime, new output/Drive paths, checkpoint cadence, backup/restore and stop/failure policy. Define evaluation before training. | User reviews the bounded plan and intended research scope. Dataset changes, if justified, need a separately versioned decision. |
| 3. Narrow research CUDA bridge | Reuse Satlas construction, CUDA timing/memory, gradients, checkpoint/RNG/backup and evaluation primitives from G1-B/G2/G3-A. Connect the existing research reader/sampler with a small reviewed entrypoint and synthetic tests. Preserve historical Zenodo10 behavior. | Small implementation PR reviewed and merged, with input/restore safeguards and unverified GPU items explicit. |
| 4. Approved CUDA smoke and run | Confirm dependencies, weights, fresh outputs and measured CUDA memory/throughput on an available T4 or other CUDA runtime. Execute the approved finite experiment and preserve failures/checkpoints. | Separate explicit execution approval and accessible runtime. Settings changes after OOM/nonfinite failure need a new decision. |
| 5. OAM evaluation, then GSI qualitative review | Keep Val18 fixed. Report per-source confusion counts, macro/micro IoU/F1, precision/recall, CE, canopy-area bias and RGB/FP/FN/probability views. Preserve undefined-denominator reasons. Once a candidate produces useful source-domain masks, inspect predefined GSI scenes. | Evidence-based model-quality and later GSI proceed/adapt/defer decisions, with commercial clearance handled separately. |

`ResearchData` and `r2_train.make_plan` are CPU-validated research supply components. `r2_train.train` enforces CPU FP32. Historical G2 CUDA is bound to Zenodo10 / 20 steps and G3-A to Val3. **The fresh v5 CUDA integration and v5 GPU experiment remain unperformed.** GEEKOM image-review access, including its Radeon graphics, provides no evidence of a usable CUDA runtime. Reuse the successful GPU primitives on a measured runtime.

This sequence has no 100-original-image quota. The six unexamined sources and 16 fully held cases stay deferred unless a documented scene/teacher gap justifies another bounded task. A complete 88-source segment crosswalk and mass Japanese reference-label collection are not entry requirements for the exploratory experiment. Research v5 and formal R1a retain distinct product/review contracts. The formal R1a/R1b acceptance road below remains applicable to production-facing acceptance.

Administrative Issues [#50](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/50) and [#56](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/56), and paused Sandbox [#64](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/64), belong to separate triage. Their disposition requires an explicit decision. They are not dependencies of the research GPU bridge.

## Historical roadmap context — 2026-11-07 and 2026-11-14

The preserved roadmap below originated with the 2026-11-07 R0–R5 plan and its 2026-11-14 priority proposal. Documentation PRs #41 and #65 were merged. Original proposal/gate language remains historical context. The dated research-only sequence above is the current next-work handoff, with separate execution and acceptance approvals.

## 2026-11-14 priority reset — focus on useful training, not compulsory isolation

This is the **proposed current work-order update** to the original 2026-11-07 R0–R5 roadmap below; the older phase descriptions remain for traceability. It follows the [risk-proportionate execution policy](DESIGN_PRINCIPLES.md#risk-proportionate-execution-policy-2026-11-14-proposal) and the 2026-11-11 [milestone review](PROJECT_REVIEW_20261111.md). The next outcome remains **a useful OAM-trained canopy model and early GSI visual inspection**, not a completed sandbox.

**Observed baseline:** Fixed 150 candidates (Train 120 / Val 30) have Technical PASS 55 (Train 49 / Val 6), HOLD 94 and FAIL 1. **Formal R1 accepted: 0**. The archived Zenodo10 are smoke-only. The 20-step GPU run demonstrated the training pipeline but predicted no canopy in the three fixed Val crops. The four-image R1d pilot (Train 717/1054; Val 1241/461) has synthetic conversion and evidence-only readiness tooling; G3/G4 remain PENDING and actual real products have not been made.

**Smallest productive next work, without GEEKOM-A6 access:**

1. **Decide the proportionate real-data path:** separate data/license/split/label/coverage requirements from self-imposed OS isolation and cumulative safety controls. Draft one narrow, reviewable host-based research execution contract if justified; keep source inputs immutable and outputs isolated. Existing `REAL_EXECUTION_DISABLED` and R1a constraints are not automatically lifted by this document.
2. **Increase usable data supply:** reuse 150-image QA; prioritize clean, diverse Train and especially Val scenes; exclude uncertain black/white coverage images and replenish candidates instead of adding mask-aware loss now. Do not promote Technical PASS to Formal R1 PASS or alter holdout/GT54.
3. **Build a generalized, resumable Trainer and synthetic/CPU tests** in ordinary development environments in parallel with data qualification. Existing G0–G3 checkpoints remain frozen.
4. **Run a bounded real training/evaluation experiment** only after input rights, split, scope, runtime and applicable execution approval are established; inspect nontrivial canopy prediction and source-domain performance, then sample GSI imagery.

**Not project-wide prerequisites:** Windows Sandbox setup/guest Python packaging, an absolute zero-byte network proof, exhaustive checks of every unknown source extent, or proving every pixel physically valid. Escalate these measures only against an identified risk. Issue [#64](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/64) is paused as an optional offline-isolation investigation, not a dependency for unrelated research/Trainer work. Its prior execution limits still apply wherever the existing R1d path explicitly enforces them, until a separately reviewed replacement is adopted.

**Scope/authority boundary:** This roadmap update is not a new license clearance, G3/G4 or `ALL_VALID` authorization, real G5/G6 permission, R1 dataset acceptance or software-execution change. Document any revised execution gate explicitly before acting on it.

## A. Outcome and priorities

**Next outcome:** Finish M1 by producing and evaluating a reproducibly fine-tuned, usable source-domain 2D canopy model; then perform an early, lightweight M2 qualitative suitability review on real Japanese GSI RGB scenes. Do not require new Japan/GSI GT or nationwide precision claims for M1/M2.

**First accepted-data expansion target: approximately 100 images total, aiming to include the existing Zenodo10 after R1 requalification**, provisionally Train 80 / Val 20. The ten historical samples are accepted for G0–G3 smoke only, not automatically for general training; retain their archived data and manifest unchanged. Prefer distinct `oam_id` sources and coarse geography/scene diversity; preserve existing fold roles. Quality/rights/split separation take priority over any numeric ratio. Accept fewer than 100 rather than admitting HOLD; modestly exceed 100 only if the approved resource budget and QA allow. No automatic use of official test/holdout or GT54. A second expansion to 200–500 remains hypothetical, not approved work.

## B. Sequenced work packages and gates

### R0 — Freeze and formalize existing evidence (documentation-only)

- Pin current G0/G1-A/G1-B/G2/G3-A states and receipts in [Current State](CURRENT_STATE.md); link from README.
- Protect existing datasets, Step10/Step20 checkpoints, G3-A result/visualization artifacts, SHA and provenance. Neither relabel nor re-run accepted results.
- **Exit gate:** reviewed documentation is internally consistent; performance `NOT MET` is separate from execution `PASS`.

### R1 — Build and accept a ~100-image dataset

**R1a: General dataset contract.**
- Define a versioned, manifest-driven schema for arbitrary accepted sample counts and image dimensions; immutable dataset identity; per-product SHA; pin Zenodo source/revision and source-byte rights; independent `oam_id`/fold/footprint checks; Tree/Canopy label and valid-area semantics.
- Keep legacy Zenodo10 verifier as a frozen regression fixture, not a target for refactoring. Test compatibility on all ten without automatically promoting their smoke-only acceptance; record separate R1 PASS/HOLD decisions against explicit label/rights/validity rules (the old manifest marks per-ID category validation `NOT_VERIFIED`). Contract must state whether training supports only verified all-valid sources initially; ambiguous validity stays HOLD until separately reviewed.
- Keep source/license assertions explicitly per sample, including attribution evidence. Do not silently infer missing/unknown row rights from archive-level claims. Preserve unaltered uniform 0/1 masks.
- **Gate:** schema and validators tested on existing ten plus synthetic multi-size, malformed, conflicting-rights, duplicate-scene and holdout examples.

**R1b: Bounded Zenodo acquisition + QA.**
- First choose a deliberately diverse candidate pool **larger than the target** using metadata and rights; avoid adjacent/repeated source regions. Record reproducible selection method and held-out separation before downloads.
- Acquire in bounded batches (e.g., 20–30 candidates), with explicit transfer/storage limits derived from pilot bytes, hash/identity checking and restart-safe accounting. Do not silently switch to historical Hugging Face sources or grant unverified rights.
- Run per-image automated checks for actual source RGB/annotation bytes, category/label semantics, GSD/affine/grid alignment, validity, unusual black-edge regions, canopy fraction and attribution.
- Begin with image-level manual review to measure automatic QA misses, then allow risk-triggered review plus a **predeclared random audit of normal cases** only after evidence justifies reduced manual intensity. A clean automated report is not proof of annotation completeness.
- Suspect sharp black borders, ambiguous validity, unknown licenses, unknown IDs and spatial cross-split contamination => `HOLD`, **not** automatic black-to-NoData conversion, cropping or acceptance. Existing HOLD remediation is separately gated if clean supply proves insufficient.
- Retain exact source IDs and avoid cherry-picking based on current Val3 model predictions or numerical outcomes.
- **Gate:** ~100 distinct-source R1 QA-accepted samples (target, not hard floor; count legacy ten only after R1 acceptance), grouped Train/Val manifest, hash-verified products and auditable PASS/HOLD reasons. Record actual accepted count and diversity rather than claiming all candidate pixels are perfect.

### R2 — Generalized, restartable GPU Trainer (can begin after R1a contract)

- Reuse verified Satlas/FPN/SEGMENT2 forward/loss/checkpoint primitives without changing architecture, loss, threshold or hyperparameters solely because G2 Step20 performance was poor.
- Separate **dataset-wide preflight verification** from per-step sample loading under an immutable staging contract; enforce source immutability across resume rather than rehash all products every optimizer step.
- Parameterize dataset manifest, crop sampling, step/epoch budget, batch/gradient accumulation when measured feasible, evaluation hooks and checkpoint cadence. Base choices on measured T4 memory/throughput; do not assume any run duration.
- Define atomic or failure-safe checkpoint publication, complete optimizer/RNG/config/dataset lineage, resumable interruption handling, and independent hash verification of persisted copies. Keep old G2 behavior unchanged.
- **Gate:** offline tests for data and resume/guardrails, bounded real T4 smoke with nonfinite checks and checkpoint continuation, recorded resource use. A passed GPU smoke is not model-quality acceptance.

### R3 — First meaningful training + source-domain evaluation

- Freeze an experiment plan **before** training: accepted dataset ID, Train/Val, seed/sampling, fixed Val assessment, optimizer and step/epoch budget from measured compute feasibility.
- Keep binary masks/probabilities and Tree IoU/F1, Precision, Recall, CE, canopy area bias; inspect per-scene and aggregate performance. Evaluate representative full images or predeclared non-overlap tiling, not only legacy three 224×224 crops.
- Maintain a leakage-free validation/holdout hierarchy: do not choose a threshold or repeatedly change settings to maximize the fixed Val3 score. The existing Zenodo10 Val3 can remain a historical regression fixture; any overlapping new Val scenes are not an independent final test.
- **Gate:** valid nontrivial canopy predictions and recorded quantitative/qualitative source-domain assessment; determine whether they are plausibly useful. IoU merely greater than 0 is not sufficient for M1 completion. If it fails, diagnose with bounded, declared changes rather than launch broad searches.

### R4 — Early real-GSI qualitative suitability review (M2)

- On several predeclared existing GSI scenes (different urban/vegetated conditions), generate geospatially alignable RGB, probability and mask results with pinned pre-/postprocessing. Review omission, shadow/building confusion, fragmentation and canopy boundary plausibility.
- GSI nominal pixel spacing is not evidence of source-native optical resolution. Check input CRS/GSD/resampling chain and model inference scale.
- No GSI labels or numeric GSI accuracy claims are required for first qualitative review; do not use GT54 as a tuning input.
- **Gate:** example images and explicit observed limitations support an informed proceed/adapt/defer decision.

### R5 — Decide next investment and OUEM handoff readiness

- If source-domain and GSI outputs are practically useful, define stable probability/mask/validity/threshold/tiling/overlap spatial contract and golden fixture for OUEM; defer downstream 3D acceptance to OUEM.
- If not useful, trace evidence to data coverage, source/target domain gap, loss/training, or architecture only as indicated. Reopen model choice or more costly labeling only for demonstrated blockers.
- Complete attribution and organizational commercial rights review before real production/commercial use; model quality acceptance never substitutes for it.
- **Gate:** documented decision and next scoped PRs; no nationwide/production readiness claim without evidence.

## C. Suggested PR boundaries

| PR | Scope | Explicit exclusion |
| --- | --- | --- |
| R0 documentation PR | Review, Current State, Roadmap, minimal README pointer | No training/data/code/weights changes |
| R1a | General manifest and validators + tests | No broad acquisition |
| R1b | Pilot batch acquisition and QA/review receipts | No model training or silent HOLD acceptance |
| R1c | Expanded dataset freeze/transfer/provenance | No overwritten Zenodo10 |
| R2a | Generic Trainer and offline tests | No long GPU run |
| R2b | T4 smoke and verified restore/cost receipts | No model-quality claim |
| R3 | Frozen real training/evaluation, acceptance report | No Val/holdout-driven parameter search |
| R4 | GSI inference/visual review | No new GSI GT requirement |
| R5 | Decision / OUEM handoff contract if warranted | No premature commercial deployment claim |

R1b and R2a may proceed partly in parallel after R1a is agreed, with integration only after relevant gates. Prefer small reversible PRs over one implementation rewrite.

## D. Success, failure and review policy

- **Execution PASS != segmentation-quality PASS != GSI accuracy != downstream OUEM/rights approval.**
- Preserve each prior G0–G3 experiment's hashes, checkpoints, source records and limitations.
- New PRs must cite measured problems, tests run, exact artifacts and explicit `NOT_VERIFIED` items. New source data, model weights and giant checkpoint files remain outside Git.
- At R1's first batch, recheck whether 100 accepted distinct sources are feasible with proportionate QA effort; reduce scope or request a reviewed remediation only if measured yield/compute requires it. The target is a plan, not an irreversible resource commitment.

See [Project Review](PROJECT_REVIEW_20261107.md) and [Current State](CURRENT_STATE.md).
