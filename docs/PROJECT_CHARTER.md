# Project Charter — OUEM Canopy Segmentation

**Status:** Proposed v0.2 design baseline (2026-11-01); becomes the active project charter when the corresponding design PR is reviewed and merged. **Not** a statement that a model, accuracy benchmark, or commercial deployment has been approved.

## 1. Why this project exists

Produce a **practical, maintainable, and commercially plausible** RGB canopy semantic-segmentation component for the **Open Urban Environment Model (OUEM)**. The intended workflow starts from accessible Japanese **GSI aerial RGB orthophotos** (typically about **0.5–0.6 m/pixel**), derives a 2D tree-canopy probability/mask, and passes it to OUEM for polygons and combination with canopy height/CHM to support simplified 3D canopy, shade/thermal and landscape applications.

**This is engineering for a usable OUEM component, not a research project aimed at demonstrating the best architecture or quantifying pretraining benefit.** Aim for **good-enough accuracy, good-enough licensing clarity, reproducibility, affordability, and an operational handoff**. The project does not require perfect segmentation, single-tree detection, a paper, a state-of-the-art benchmark, or exhaustive model comparisons.

**Milestone discipline (confirmed 2026-11-01):** The immediate milestone is to **complete a working SatlasPretrain + OAM-TCD CC BY 4.0 fine-tuning workflow** and generate canopy masks. Do not impose GSI reference-label creation, independent Japanese quantitative accuracy thresholds, exhaustive benchmarking or downstream environmental validation on this first milestone. Next, inspect predictions **visually across several existing GSI environments**; decide further action only from observed problems. A successful training workflow is not evidence of nationwide or production-grade accuracy. Quantitative Japan/GSI evaluation may be added later if a concrete need emerges.

## 2. Fixed scope / boundaries

- **Input:** RGB aerial imagery. OUEM owns GSI acquisition, native CRS/georeferencing, valid-data masks, large-area tiling and geographic outputs. Model repository owns RGB preprocessing and model input contract.
- **Output and acceptance boundary (confirmed 2026-11-01):** This repository produces and validates **2D Tree/Canopy probability maps and binary masks** from GSI RGB imagery, with reproducible inference metadata and spatially alignable outputs. **The canopy mask is an intermediate layer, not the environmental-assessment product.** OUEM owns georeferencing, polygonization, **integration with point-cloud data to construct canopy height-distribution/3D canopy data**, and subsequent shade/thermal/landscape evaluation. The effectiveness and errors of point-cloud height assignment and environmental assessment are **OUEM-level acceptance responsibilities**, not acceptance requirements for this segmentation repository. The segmentation handoff must preserve the grid, mask continuity and positional alignment needed for point-cloud integration.
- **Target scale:** primarily GSI ~0.5–0.6m/px. OAM-TCD nominal ~0.1m/px imagery is **training source**, not the operational input resolution.
- **Permitted source data:** **only the documented CC BY 4.0 portion** of OAM-TCD for training, subject to dataset-revision, per-record provenance, label QA and attribution checks. No automatic inclusion of separately licensed NC/SA records.
- **Model distribution:** **no public release or redistribution** of the trained OUEM model/checkpoint is planned. Intended future usage includes **internal fine-tuning and potentially commercial inference**. No rights claim is made for a different deployment or redistribution scope.
- **Not in initial scope:** instance/single-tree delineation, scientific pretraining-vs-scratch ablations, exhaustive model search, direct use of ImageNet images for commercial training, new large-scale Japanese annotation, or nationwide-performance claims.


**Target GSD decision (deferred):** Do not lock the fine-tuning target to 0.5m or 0.6m yet. Before implementing M1 resampling, inspect the existing GSI GeoTIFF CRS, pixel spacing in ground units, source tile zoom and reprojection/resampling chain. Distinguish Web Mercator pixel spacing from local ground resolution and from source-photo native detail. Choose one practical OAM-TCD training GSD on that evidence; no mandatory 0.5m-versus-0.6m experiment.

## 3. Preferred construction route (engineering decision, not a demonstrated result)

Use the Ai2 **SatlasPretrain aerial single-image RGB checkpoint `Aerial_SwinB_SI`** as the preferred initialization. Start with the **official pretrained backbone + optional pretrained FPN**, attach a **new binary segmentation head/decoder** consistent with official library semantics, then fine-tune using the **OAM-TCD CC BY 4.0** imagery and verified binary Tree/Canopy masks, transformed to approximate GSI resolution.

The exact head selection, dimensions, valid-mask/loss semantics, whether FPN is beneficial, fine-tuning freeze policy and optimization settings **remain to be inspected and tested**. This document does not prescribe a finished implementation or guarantee a training run.

The initial path is deliberately **one engineering candidate**, not a three-arm research study:
- **Preferred:** Satlas-pretrained Swin-v2 Base + binary semantic head + OAM-TCD CC BY-only fine-tuning.
- **Fallback only if blocked:** evaluate a smaller independently permissible model, changed decoder/learning setup, or additional licensed Japanese samples according to measured blocker. A Swin-v2 Base scratch comparator is **not required**.
- Previous released Restor OAM-TCD pretrained checkpoints remain **excluded from a commercial-candidate weight chain** under currently documented noncommercial/restricted terms. The **OAM-TCD labels/data remain central**.

## 4. What is established versus what is open

| Topic | Established/documented | NOT YET verified / decision still needed |
|---|---|---|
| Operational input | GSI RGB around 0.5–0.6m/px; prior local scale tests show site-dependent behavior | Robust generalization over Japanese sites, seasons, quality and landscapes |
| Satlas pretrained checkpoint | Ai2 lists `Aerial_SwinB_SI` for ~0.5–2m RGB and explicitly designates its pretrained weights **ODC-BY**; loader code Apache-2.0 | Exact downloaded artifact SHA/revision; actual model import, input/output, dependencies and GPU memory |
| Upstream ImageNet | Published Satlas training path initializes a torchvision Swin-v2 Base ImageNet checkpoint; original ImageNet data access has noncommercial terms | Legal treatment of downstream checkpoint rights and weight-specific conditions; commercial production review |
| OAM-TCD | Provider publishes a CC BY-only partition and label lineage; original study used high-resolution imagery and pretrained baselines | Actual chosen revision/rows, raw image and panoptic mask verification, attribution completeness, data quality at target resolution |
| Preprocessing | ~10cm→50cm controlled 5x degradation is documented; area RGB and fractional-mask aggregation planned | Whether initial operational training should target ~0.5m, ~0.6m or both; best crop/grid policy; resampling/label boundary impact |
| Evaluation | Tree IoU/F1, precision, recall, canopy area bias, shadows/roads/buildings, small canopies and edge quality are meaningful | Quantified minimum acceptable performance, independent Japanese GT availability, inference throughput and OUEM downstream tolerance |
| Geographic evaluation | Earlier GSI visual trials at Hanamigawa / Aobanomori exist | No numeric GSI accuracy claim; **GT54 land-cover labels are not automatically valid canopy GT** and GT54 must not be tuned on |

Research evidence lives in `docs/research/` including the merged P0/PR #7 provenance records. **Documentation evidence is not a completed runtime test, rights opinion or model validation.**

## 5. Sequential decision gates (do not claim PASS early)

### D0 — Mission/design rebase (this stage)
- Clarify purpose, preferred route, constraints, evidence levels and the superseded experimental plan.
- Review and merge the **documentation-only** PR before instructing an implementation agent from the new source of truth.

### D1 — Minimal official implementation + rights manifest (before data training)
- Freeze package/source revisions and check official `Aerial_SwinB_SI` loading, normalization, model architecture (backbone/FPN/head), forward/loss/target shapes and Windows/CPU compatibility on synthetic input.
- Inventory direct and transitive code licenses and weight attribution. Record Ai2's explicit ODC-BY plus unresolved upstream ImageNet issue; no claim of legal signoff.
- Plan measurable GPU preflight (memory, patch/batch sizing). **Do not assume RTX A2000 is sufficient without a run.**

### D2 — CC BY-only data QA
- Pin OAM-TCD source/revision, per-row licenses and credits, exact Tree/Canopy mapping and background/void handling.
- Verify actual RGB/mask pairing and grid alignment, deterministic `oam_id`-group split, source/output hashes, and no holdout/GT54 leakage.
- Choose and document one **initial** target-resolution transform; keep 5x (~0.5m) from the previous plan as a defensible starting point, not a physically exact 0.6m equivalence.

### D3 — M1: complete the Satlas + OAM-TCD fine-tune
- CPU/synthetic smoke, GPU micro-step, limited real-data overfit and held-out OAM-TCD validation, then a bounded main training run **only if earlier gates pass**.
- **M1 completion:** verified CC BY-only dataset/preprocessing, reproducible Satlas-initialized training, saved locally usable checkpoint and successful canopy-mask inference, with basic source-domain validation and run provenance. No mandatory scratch control, new Japanese GT, or quantitative GSI accuracy gate.
- Log initialization, full configuration, loss behavior, throughput, VRAM, deterministic checkpoint behavior and basic validation metrics. These are diagnostics, not proof of operational generalization.

### D4 — GSI visual suitability review (after M1, initially lightweight)
- Run inference on several readily available GSI scenes/environments and inspect canopy omission, shadow/building false positives, fragmentation and scale plausibility. Record example images and observations, **without requiring new GSI ground truth**.
- This is a **qualitative check**, not a numeric GSI accuracy certification or a nationwide generalization claim. Do not invent metrics without reference labels. Preserve GT54 independence.
- Only if observed shortcomings or downstream requirements justify the cost, propose additional GSI annotations, targeted training, wider systematic evaluation or a fallback. Do not precommit to such work.

### D5 — OUEM handoff/operational readiness
- Version and checksum a local model package, ship with preprocessing/threshold/GSD/output semantics, golden inference fixture, and spatial contract; validate downstream seams and polygon/CHM integration.
- Separate **model quality acceptance** from **organizational commercial rights review**. Keep attribution and use-scope records. Do not claim broad nationwide accuracy without evidence.

## 6. Quality/decision discipline

- Preserve provenance and reproducible training/inference artifacts. No hidden training on official OAM holdout or GT54.
- Do not confuse training/validation score, visual plausibility, independent Japan accuracy, and downstream OUEM suitability.
- Prefer a bounded, functional engineering prototype over a research-perfect benchmark; however accuracy and licenses are **gates**, not casually waived.
- Current model selection is an **approved direction to evaluate**, not an assertion of adequate quality or final commercial clearance.
- Change one meaningful design decision per reviewable PR, note consequences and rollback options, and keep earlier studies in Git history and `docs/research/`.

## 7. Document precedence

1. This **charter**, upon merge, controls project purpose, scope and high-level decisions.
2. [Design principles](DESIGN_PRINCIPLES.md) control mandatory engineering constraints.
3. [ADR: Satlas-first rebase](decisions/ADR_20261101_SATLAS_FIRST.md) explains trade-offs and supersession.
4. Future **active** implementation plan (to be written as a separate reviewable change) controls commands/training details.
5. [Legacy experiment plan](EXPERIMENT_PLAN.md) and earlier research docs remain historical evidence **when they conflict with this approved direction**.
6. [OUEM handoff](OUEM_HANDOFF.md) remains a preliminary responsibility/API contract pending later revision.
