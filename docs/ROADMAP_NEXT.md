# OUEM Canopy Segmentation — Next Roadmap

**Revision:** 2026-11-14 priority update proposed for review; underlying R0–R5 plan originated 2026-11-07  
**Authority:** subordinate to [Project Charter](PROJECT_CHARTER.md) and [Design Principles](DESIGN_PRINCIPLES.md). The existing M1/M2/D0–D5 scope is unchanged. R0–R5 below describe the next engineering **work packages**, not new, overriding scientific acceptance standards.

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
