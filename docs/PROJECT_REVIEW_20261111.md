# OUEM Canopy Segmentation — Milestone Review (2026-11-11)

**Status:** REVIEW PROPOSAL / NOT YET APPROVED. Documentation checkpoint at the end of R1d-1B-4; no data acceptance, execution approval, rights sign-off, or model-quality certification is implied.

**Review disposition proposed:** **CONDITIONAL PASS — retain the Satlas-first direction, shift the next phase from evidence accumulation to bounded implementation, accepted data, and meaningful training.**

## 1. Mission and precedence

The governing goal is a practical, reproducible 2D RGB Tree/Canopy probability and mask component for the Open Urban Environment Model (OUEM), primarily using Japanese GSI aerial imagery at roughly 0.5–0.6 m/local-ground pixel. OUEM itself owns geospatial ingestion, 3D canopy/point-cloud integration, and downstream shade, heat and landscape assessment.

The first milestone **M1** is a functioning, verified OAM-TCD/Satlas fine-tuning workflow with meaningful source-domain canopy masks, a saved checkpoint, and basic evaluation; **M2** is early qualitative GSI visual review. Neither requires new Japanese ground truth, an exhaustive architecture comparison, or nationwide accuracy claims.

This document is subordinate to [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md), [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md), and the applicable R1a implementation contract. It supplements—does not replace—the [2026-11-07 review](PROJECT_REVIEW_20261107.md), [Current State](CURRENT_STATE.md), and [Next Roadmap](ROADMAP_NEXT.md). Issue proposals are not presumed to amend accepted contracts.

## 2. Verified or reported progress by 2026-11-11

| Area | Evidence/state | Limit |
| --- | --- | --- |
| Model approach | Ai2 SatlasPretrain `Aerial_SwinB_SI` backbone/FPN plus new official two-class SEGMENT head remains the preferred route | No reason yet to change model family |
| GPU engineering | Tesla T4 bounded training, checkpoint/restore and fixed Val3 evaluation executed | M1 model quality **not met**; Step20 predicted all background on three fixed crops (micro Tree IoU/F1 = 0) |
| R1b candidate pool | Exactly 150 fixed candidates: Train120 / Val30; QA: **Technical PASS55, HOLD94, FAIL1** | Technical PASS is not Formal R1 acceptance |
| R1c coverage investigation | Black/white coverage provenance investigated; A+D (exclude/hold uncertain candidates and replenish clean ones) preferred | Black/white pixel values alone are not validity semantics; no valid-mask/loss redesign approved |
| R1d-0 readiness | Technical PASS55 = Train49 / Val6; six evidence gates inventoried | All six were partially evidenced at the outset; missing evidence is image-specific |
| R1d-1A criteria | G1–G6 source/rights, independence, coverage, 0.50m product and reproducibility requirements proposed | Proposal is not automatically a modified R1a contract |
| R1d-1B-1 to -3 pilot | Fixed Train 717/1054 and Val 1241/461 reviewed. G1/G2 document evidence judged sufficient; G4 lightweight review technically supportable; G3 scope interpretation remains | No independent user final acceptance. Four remain HOLD |
| R1d-1B-4 plan | Bounded G3/G4 application proposal and 4-image G5/G6 execution/safety plan recorded | **G5/G6 not run**; existing CLI is not safe to execute as-is against proposed constraints |

**Current formal R1 accepted count in the fixed 150-candidate pool: 0. All 150 remain Formal R1 HOLD.** The archived Zenodo10 are accepted for smoke only and are not promoted to general R1 training acceptance.

Issue-specific evidence and reported hashes live in [Issue #52](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/52) and [Issue #53](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/53); in particular, the [R1d-1B-4 report](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/53#issuecomment-6771685283). These are report-level review sources: this documentation change does **not** independently rehash local raw imagery, 31 diagnostic PNGs, ZIP/archive receipts, or outputs.

## 3. Decisions to preserve

1. **KEEP** Satlas-first / official backbone + FPN + segmentation head. The 20-step all-background result on a tiny training set is not proof the architecture is unsuitable.
2. **KEEP** near-0.50m area-based RGB/fractional-label preprocessing as a documented engineering candidate, not as proof that synthetically degraded OAM imagery reproduces native GSI imaging.
3. **KEEP** source-level CC BY-only rights evidence, attribution, exact source/annotation hashes, fold and source-group separation, designated holdout and GT54 independence, and clear separation between technical QA, evidence sufficiency, formal acceptance, and actual model quality.
4. **KEEP** A+D data expansion by default. Source coverage is not inferred from black/white colors, from STAC bbox containment, or from a nominal `all_valid` GDAL mask alone.
5. **KEEP** approved legacy G0–G3 receipts, source data, checkpoints, Campaign/Registry/Approval and older run states immutable. Never silently rewrite or upgrade smoke-only acceptance.

## 4. Remaining approval and evidence boundaries

### G3: bounded external independence

The pilot used a fixed official Train/Test catalog (4,169 / 439 metadata, 541 groups), available source metadata including 17 known STAC bboxes, object-path aliases, and the existing 1 km proximity guard. No known cross-split leakage was detected for the four pilot images.

**Residual risk:** the complete source scenes for 524 groups, unknown aliases, and catalog-external scenes were not exhaustively established; group envelopes are not original scene footprints. The limited external scope, catalog/index identity and those limitations must be explicitly accepted for the four images and current internal research use before G3 can be considered cleared. No blanket approval for other images is implied. Do not require an unbounded search of all unknown imagery, but do not convert a large nominal distance into a universal independence guarantee.

### G4: operational coverage review versus physical proof

For the four pilot images the existing whole-image RGB/label overlays, bounded full-resolution windows, black/white flags and source-bound receipts reported no significant unresolved synthetic border or large alignment error in the reviewed areas.

The proposed lightweight whole-tile review can be accepted as a bounded operational judgment, **not** proof that every pixel is physically observed. The existing adapter requires `review.coverage=ALL_VALID` and emits all-one `valid_fraction`; interpreting this token as an approved operational all-valid assertion requires explicit, recorded authorization and verification of compatibility with the authoritative R1a contract. `REVIEW_PASS` must not be automatically rewritten to `ALL_VALID`. This approval is not yet issued.

### Rights and semantics

Existing source-bound rights receipts for the four pilot images were judged sufficient for current internal research documentation; eventual commercial inference, redistribution, starting-weight/ImageNet chain, and organizational sign-off remain separate. Semantic correctness of Tree/Canopy positives, panoptic category/segment mapping and small-tree completeness still warrant bounded verification where relevant, especially before wider training.

## 5. Engineering constraints uncovered

The existing Zenodo converter cannot be run directly under the proposed R1d-1B-4 safety contract: it catches per-image errors and continues, mutates a manifest, writes products/previews directly, and does not enforce the requested cumulative write and hard memory ceilings.

The four-image plan proposes RGB uint8, canopy_fraction float32, canopy_mask uint8 and valid_fraction float32 on a shared near-0.50m grid, with repeat generation in a separate output and pixel/grid/readback/SHA comparison: 4 images × 4 products × 2 executions = 32 GeoTIFFs. Proposed ceilings: 64 MiB total new writes (including partials/repeats/reports), 2 GiB hard worker memory, 1,800 s worker runtime, zero network. These are **proposed limits, not verified enforcement**. The code shim, bounded writer, memory enforcement and regression tests are not yet implemented.

An approved R1a full preflight PASS cannot be manufactured before genuine `r1_review` and final human approval. Review-ready products and formal dataset acceptance are separate milestones.

## 6. Scientific and operational risks, ranked

| Priority | Risk | Mitigation / next evidence |
| --- | --- | --- |
| High | No accepted expanded Train/Val and therefore no meaningful new GPU run | Finish the bounded four-image data path, then scale evidence reuse and accepted dataset preparation |
| High | Only six Technical PASS Val images; five have source-resolution teacher canopy fraction below 10% | Select/replenish Val for geographic, scene and canopy-fraction diversity using predeclared source/split rules, never model scores |
| High | Panoptic-to-binary target semantics and fractional labels at near-0.50m may omit small canopies or mis-map segments | Bounded semantic checks; preserve full-zero/full-one labels and original area-weighted definitions |
| Medium | GSI domain shift: resampled high-resolution OAM is not native low-resolution GSI photography | Real GSI qualitative review after a usable source-domain model; no premature Japan accuracy claim |
| Medium | GPU general Trainer still Zenodo10/20-step-specific | Build variable-manifest, restartable, checkpoint-safe Trainer in parallel once R1a contract is stable |
| Medium | Over-engineering of audits and serial approval steps | Reuse common SHA-bound receipts, predeclared risk-tier image review, compact reviewable PRs and decision gates |
| Later | Commercial deployment and OUEM 3D/thermal outcomes remain unreviewed | Separate rights and downstream integration gates, not M1 acceptance prerequisites |

## 7. Next phase and exit gates

**Phase B1 — Minimal safety implementation and synthetic tests, in parallel with G3/G4 decisions.** For the fixed four pilot images only, independently review and record decisions on G3 bounded-scope residual risk and G4 adapter `ALL_VALID` semantics. The plan-bound shim / bounded writer, strict fail-fast, atomic publication, write/memory/time/network controls and backward-compatibility tests may be implemented and tested **on synthetic inputs in parallel**; these tasks do not constitute G3/G4 acceptance. **Real imagery product generation is prohibited until the G3/G4 decisions, implementation and safety verification, and separate execution approval are all complete.**

**Phase B2 — Four-image 0.50m product verification.** This four-image pilot validates the acceptance and product-generation **workflow**; it does **not** establish a sufficiently diverse dataset for meaningful model training or validation. Once explicit G3/G4 decisions, execution approval and validated safety enforcement exist, generate the 4-product set and independent repeat, verify input/output SHA, geospatial grid, fractions, mask thresholds, readback and determinism. Review the candidate manifest and evidence before any final R1 acceptance.

**Phase C — Expand eligible data.** Use the common archive/source receipts and targeted per-image quality review to expand Train/Val; replenish especially diverse Val candidates. The existing approximately 100 accepted-image target (provisionally 80/20) is a planning guide, **not** a pass-count mandate. Do not auto-promote the 55 Technical PASS samples or historical Zenodo10.

**Phase D — Generalized Trainer and meaningful source-domain evaluation.** R2 offline work may proceed in parallel with R1 data supply. Freeze dataset/Train/Val, optimizer, seed, evaluation and GPU budget before a run; require nontrivial canopy predictions and reproducible checkpoints. Follow with lightweight GSI qualitative suitability (M2), then only problem-driven further work.

**Next outcome-based checkpoint:** the first defensible, accepted Train/Val and a reproducibly fine-tuned model with nontrivial source-domain canopy predictions. Merely finishing another audit or getting IoU > 0 is not enough.

## 8. Change and review boundaries for this milestone

This record is a documentation-only review proposal. It **does not** approve the G3/G4 application, alter R1a/adapter semantics, authorize G5/G6 execution, generate product files, modify QA or manifests, perform GPU work, change an accepted dataset or authorize commercial deployment.

If accepted, keep this historical checkpoint immutable and reference it from subsequent PRs. Future implementation must separately pin its source commit, exact input hashes, affected files, safety tests, residual risks and user approval. The operational status should be updated in a new record after execution rather than retroactively turning this proposal into a successful run.
