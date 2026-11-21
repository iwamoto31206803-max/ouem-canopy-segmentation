# D6-A: Phase C / R1b QA-dependency audit and objective admission design

**Date:** 2026-11-21 (JST)  
**Status:** **PROPOSAL / INVENTORY ONLY**. Source code and private data have **not** been changed or rerun for this audit. The decisions below are **D6 exploratory research-admission proposals**, not Formal R1 admission, legal clearance, annotation correction or a retrospective reclassification of Phase C artifacts. See [D6 plan](../D6_PREPROCESSING_GPU_OPERATIONS_PLAN.md).

## 1. Clarified owner objective: objective decision-making, not byte-perfect cross-PC output

The primary goal is **automated high-volume source -> preprocessed patch -> rule-based eligibility/QA -> frozen research dataset**, with **no mandatory human or Codex per-image visual judgement** to decide whether an ordinary sample proceeds. An identical, versioned policy and the same source evidence should lead to the same **selection, PASS/HOLD/FAIL reason and crop recipe**. This is **decision/procedure reproducibility**, not a requirement to prove byte-identical TIFFs on two operating systems.

The human owner approves the **policy and evidence boundaries once** and may evaluate a separate stratified audit sample to assess errors or revise the next *versioned* policy. Human/Codex inspection cannot silently promote individual images, alter recorded teacher labels or become a required gate for each candidate. A rule whose only implementation is "looks acceptable" is **not an automatic QA rule**.

## 1A. Owner decision: trust the published OAM-TCD labels as research input

**Approved scope principle (2026-11-21):** OAM-TCD is an officially produced, published annotated dataset. For **exploratory training**, treat its upstream tree/canopy annotations as the **default source of teacher truth**, accepting that quality and interpretation are not fully uniform. **Do not make human/Codex visual review, per-image label-completeness checks, or nursery/shrub/tree boundary adjudication prerequisites for ordinary research admission.** This does not assert that every annotation is correct.

**Keep the automatic rejection surface minimal and defensible:** source/dataset identity and license declarations, allowed official fold and source-group separation, malformed input/mismatched dimensions/CRS, corrupt hashes, declared invalid pixels or clearly calculable unusable crop support. Use the existing deterministic black-support crop avoidance as a **geometric heuristic**; do not infer that all dark/white pixels or black-adjacent crowns are wrong. Preserve original source imagery/annotation and all-zero/all-one masks, with no automated label correction.

**Metadata and rights are independent of visual trust:** record and verify the documented licensing/attribution evidence under a reviewed source policy. Missing or contradictory required evidence causes machine-reasoned HOLD, not per-image discretionary visual approval or a claim of commercial legal clearance.

**Scientifically transparent limitation:** record at the *dataset/run level* `UPSTREAM_OAM_TCD_LABELS_ACCEPTED_AS_IS_FOR_EXPLORATORY_RESEARCH` and `ANNOTATION_SEMANTIC_ACCURACY_NOT_INDEPENDENTLY_VERIFIED`. A **separate optional stratified sample audit**, or downstream controlled model assessment, may identify systematic teacher issues and motivate a new version of the policy; it must not become mandatory manual clearance of every image.

**Next implementation:** [D6-A1 Issue #103](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/103). Its initial code/test scope is deliberately narrow: no new real-data batch, no GPU training, no source repair, no Formal R1 status change.

## 2. Evidence-based inventory of existing behavior

| Existing condition / actual code | Machine-determinable? | Existing limitation / D6 disposition |
| --- | --- | --- |
| Source `image_id`, `oam_id`, `official_split`, fold, declared `CC-BY 4.0`, allowed source generation | **YES** — `r1b_candidate_plan.select`, `m1_5d_zenodo_prepare.select_pilot` | Preapproved **source-policy** is necessary; metadata declaration is not independent legal certification. Exclude official test/holdout; hold missing/conflicting declarations. |
| Source-bound STAC: ID, row/image license, named producer/licensor, original metadata object key, bbox | **YES for documentary consistency** — `m1_5b_pilot.rights`, `r1b_qa_runner.check_stac` | Missing or conflicting evidence causes rule HOLD; declare what documentary claims are actually supported. Do not label automated declarations as legal clearance. |
| Input hashes / archive member identity / TIFF/PNG format, RGB bit depth, dimensions, north-up affine, CRS, masks/nodata, grid/bounds | **YES** — `r1b_qa_runner.run_one`, `m1_5d_zenodo_prepare.screen` | Existing reject/hold reasons are usable; do not automatically recode grayscale to RGB or invent valid pixels. Distinguish malformed inputs (FAIL/EXCLUDE) from incomplete documentary evidence (HOLD). |
| Labels: positive upstream nonzero panoptic RGB; COCO category-set IDs in {1,2}; preserve all-zero/all-one | **YES for encoding/metadata** — `m1_5d_zenodo_prepare.semantic_binary`, `r1b_qa_runner.run_one` | **No complete segment-ID/category crosswalk** is established by the union decoder. Preserve `category_crosswalk_status=NOT_VERIFIED`; if approved research source policy permits upstream union with this known limitation, record it explicitly, not as segment-level proof. |
| Metadata split independence: canonical `oam_id`/original object, footprints, known opposite-role proximity, source aliases visible in catalog | **YES inside defined catalog** — `m1_5b_pilot.separation`, `r1b_candidate_plan.select` | No claim about unlisted nearby/overlapping originals. Record catalog scope/hash and keep strict holdout separation. Set policy for missing/near/duplicate facts before selection. |
| Exact edge-connected black and near-black/white support diagnostics | **YES as measurements** — `m1_5b2_border_audit.component_report` and `r1b3_scope`/Runner | **NOT YES as semantic validity classification.** Shadow, asphalt, genuinely dark crowns and nodata cannot be reliably distinguished by RGB black alone. |
| R1b technical HOLD when all-valid raster mask fails or 4-connected exact border-black fraction >= 1% | **YES as current heuristic** — `r1b_qa_runner.run_one` | The 1% threshold is a **pre-existing rule, not a scientifically established nodata boundary**. Keep as a versioned candidate policy and measure yield/lost useful pixels; don't generalize as a guarantee. |
| Research crop: exclude windows intersecting edge-connected exact-black source support; choose among row-major candidates via `random.Random(f'{seed}:{image_id}')` | **YES** — `research_patches.choose_window`; `blocked_grid`/area aggregation | Already avoids per-image visual selection; no clean crop -> rule HOLD/SKIP. **White support currently diagnostic only**, not universal invalidity. Keep uniform tree/background masks, do not tune by Step280 predictions. |
| RGB area resampling; fractional canopy, >=0.5 mask; GeoTIFF readback, shape/CRS/affine/hash | **YES** — `m1_2d_prepare_pairs`, `research_patches` | Procedural calculation reproducible subject to recorded environment and numerical tolerances; no per-image human decision required. Full source array + dense `reduce_area` matrices may become a RAM bottleneck at scale; optimize after fixing regression fixtures. |
| Source/project quality: tree versus shrub/crop/nursery, missing trees, annotation alignment at crown edge, artificial white margins, true extent outside raster masks | **NO reliable automatic truth from present inputs** | Phase C reported these through **Codex AI image screening** and `research_review.json` `CLEAR_FOR_EXPLORATION`/HOLD; `ResearchData` **requires** an explicit review decision today. D6 must remove that per-image visual decision from its *exploratory* admission contract, rather than silently claiming the ambiguity resolved. |
| Formal R1 `category_review`, `validity_review`, `independence_review`, `external_independence_review`, `r1_review` | **NOT auto-promotable under current contract** | `r1_dataset` and `R1A_DATASET_CONTRACT` require source-bound explicit assertions. Formal R1 remains HOLD. D6 uses a **distinct versioned research-only contract**, never spoofs human review fields or upgrades Formal R1 automatically. |

Source basis: [Phase C1](PHASE_C1_RESEARCH_PATCHES.md), [Phase C2](PHASE_C2_RESEARCH_INPUT.md), [Phase C3](PHASE_C3_RESEARCH_POOL.md), [Phase C4](PHASE_C4_RESEARCH_DATASET.md), [R1b QA Runner](R1B2B_QA_RUNNER.md), [R1b3 scope](R1B3_SCOPE.md), [Formal R1a](R1A_DATASET_CONTRACT.md), and `scripts/{research_patches.py,research_data.py,m1_2d_prepare_pairs.py,m1_5d_zenodo_prepare.py,m1_5b2_border_audit.py,r1b_qa_runner.py,m1_5b_pilot.py}`. They are the actual implementation/dated records, not claims of a new implementation.

### Important historical example

Phase C2's `1054` sample was placed in `EXCLUDE_FROM_FIRST_TRAINING; LABEL_REVIEW_REQUIRED` after image/label observation despite an all-zero teacher. C3 withheld `4315` because of white artificial margins and also retained some roofs with white support. C4 withheld `3986` for ambiguous black holes. These are rational **case-specific research observations**, but not a globally reusable, wholly objective pass/hold rule. An all-background teacher or white pixel must not by itself be an automatic rejection rule.

## 3. Proposed D6-A research policy contract

Declare a distinct immutable `d6a-research-policy-v1`, with named rule IDs, parameters, version, pinned input/source catalog and a **machine-generated** source/window decision report. The policy is approved before looking at the next model's predictions or reselecting inconvenient samples.

### Decision levels (different guarantees)

1. **`AUTO_ELIGIBLE_RESEARCH`**: all required **measurable technical, documentary and split/sampling rules** pass under the chosen research policy, with source/product integrity checked. **This status means eligible for a bounded exploratory training dataset only**, not verified canopy truth, full true image coverage, independent legal review, or Formal R1.
2. **`AUTO_HOLD`**: source/documentary evidence missing or contradictory; explicit nodata/mask, insufficient trustworthy candidate window, unknown category semantics under the adopted policy, high-risk geometry or source-identity issue, memory/time cap, etc. Every hold must cite a **specific machine-evaluable rule ID and observed values**, not `looks odd`, `Codex unsure`, or unspecified `visual review pending`.
3. **`AUTO_EXCLUDE` / `TECHNICAL_FAIL`**: prohibited official test/holdout or license, irrecoverably unsupported format, corrupted/mismatched source bytes, invalid geometry or an execution/integrity failure, as appropriate. Do not conflate a rejected sample with a failed batch operation.
4. **`UNASSESSED_SEMANTIC_RISK`** (orthogonal flag): the genuine tree-vs-shrub, omissions, annotation completeness and unknown original-coverage questions remain **not certified**, regardless of AUTO_ELIGIBLE. Never convert a missing claim into a PASS receipt.

### Rules to decide before coding (candidate, not final thresholds)

- **Rights/source policy:** accepted provider dataset revision and publicly declared metadata/license evidence, permitted source IDs/folds, categories and the documented **upstream binary-mask union** with an explicit statement of its segment-crosswalk limitation.
- **Split/leakage policy:** group by original scene/source identity; check known catalog and protected holdout; define whether 1km near-opposite-role heuristic is a HOLD or exclusion and how catalogs are frozen.
- **Coverage policy:** **prefer authoritative validity masks/footprints when available**; otherwise use a conservative, quantitative geometry/connected-component rule to avoid *suspected* border-connected exact-black support in candidate windows. Record that this is a heuristic, not certainty. Treat exact-white and near-black as recorded diagnostics until a predictive, prespecified numeric rule is justified; no arbitrary per-image white/black human promotion.
- **Patch policy:** source-balanced deterministic crop counts (not chosen by model performance), integer `seed:image_id` keys, fixed evaluation sampling, duplicate-window and Train/Val checks, full/empty teacher masks preserved; no post-hoc selection by IoU or crown coverage.
- **Quality policy:** no `minimum canopy fraction`, `teacher looks correct` or human/Codex CLEAR requirement by default. A representative **stratified sample audit** may assess semantic noise and prompt a new policy version; it does not individually gate the entire production line.

**Human role:** approve/revise policy and legal scope **at dataset/source-policy level**, inspect aggregate QA yield and optionally a **separately sampled** set of images to estimate error frequency. An auditor may flag a material systematic issue that halts and versions the policy; cannot secretly amend individual eligibility without a recorded new rule and reprocessing.

## 4. Proposed first scope: policy and synthetic regression, not mass data processing

**A0 — Policy inventory / review (this document):**
- Verify historical conditions and remove visual-gate assumptions from the D6 plan.
- Specify status/reason schema: `rule_version`, `source_id`, `crop_xywh`, `input_sha256`, `rule_id`, `observed`, `threshold_or_expected`, `decision`, `limitations`; aggregate counts by reason.
- Owner reviews the **coverage**, **upstream teacher** and **research-only eligibility** risk tradeoff before they become executable acceptance rules.

**A1 — Deterministic decision engine + tests (later reviewed code PR):**
- Separate source metadata/rights preflight -> product conversion/crop-selection -> computed eligibility -> immutable report; **no `research_review.json` human/Codex per-image CLEAR input** for this *new* research path.
- Synthetic fixtures: exact-black border wedge versus internal dark area; legitimate white roof; declared nodata; absent/contradictory STAC; grayscale TIFF; bad source hash; no clean 224-pixel window; zero and all-one teacher; duplicate/near-source across splits; shuffled process order.
- Make output/status independent of batch order and repeatable with unchanged inputs/policy. Treat different-PC repeat as a **small smoke check**, not a blocking byte-identity/cross-OS equivalence project.
- Protect existing C1–C8/v5/G0–G3, `ResearchData` legacy path and `r1_dataset` Formal R1 contracts; no automatic migration of existing CLEAR or HOLD cases.

**A2 — Bounded real-data dry-run and yield audit (separate approval):**
- Use a small predeclared, legally eligible **fixed** input set and frozen source metadata (no hidden network or newly invented rights).
- Report `AUTO_ELIGIBLE_RESEARCH`, `AUTO_HOLD`, `AUTO_EXCLUDE` by rule, eligible **unique source count**, crop count, source-balanced diversity, overlap, RAM and CPU time. A high HOLD rate leads to **review of rule design and source quality**, not blind loosening or per-image visual overrides.
- Compare proposed objective status against past Phase C observations for diagnostic **disagreements**, not to tune rules image-by-image.
- Only after evidence and review, design idempotent batch execution/resume and larger research dataset freeze; no direct Formal R1 promotion.

### Proposed D6-A gate (not yet PASS)

`OBJECTIVE_RESEARCH_ADMISSION_POLICY_REVIEWED` + `NO_PER_IMAGE_VISUAL_GATE_TESTED` + `RULE_REASON_AND_FAIL_CLOSED_TESTS_PASS` + `BATCH_YIELD_AND_IDEMPOTENCE_MEASURED`. Cross-PC pixel identity / TIFF byte equality is **not a primary gate**. Retain ordinary input/output hash, spatial integrity and versioned environment records.

## 5. What is deliberately not decided here

- Numeric parameters for new black/white support, source proximity, missing-coverage tolerance and acceptance ratios: **measure and justify** before adoption; current >=1% R1b border-black flag is a recorded historical candidate, not automatically a D6 policy.
- Whether a fully verifiable validity footprint is available for every permitted source; pixel-only heuristics cannot certify all-valid photographic coverage.
- Whether label semantic noise would be acceptable for an expanded research run. No manual per-image gate is required, but independent sample quality studies can reveal a need to change data source or policy.
- No changed data, future model accuracy, rights certification, full-scale GPU plan, or Formal R1 acceptance.

**Implementation direction:** maximize mechanically eligible, licensed and geometrically defensible training data while clearly marking semantic uncertainty. Human review belongs in **policy design and sample audits**, not in every source/patch admission transaction.
