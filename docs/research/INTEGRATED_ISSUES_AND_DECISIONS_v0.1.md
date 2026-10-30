# OUEM Canopy Segmentation — Integrated Issue & Decision Review (v0.1)

**Status:** Research synthesis / proposed decision gates, **not architecture approval or commercial legal clearance**  
**Date:** 2026-10-31  
**Scope:** Consolidates [OAM_TCD_AUDIT_01.md](OAM_TCD_AUDIT_01.md) through [OAM_TCD_AUDIT_06.md](OAM_TCD_AUDIT_06.md). Existing experimental plans remain hypotheses pending the gates below.

## 1. Purpose and non-negotiable requirement

Build a maintainable pipeline to derive nationwide-relevant 2D canopy coverage from accessible aerial RGB imagery, then connect to height/CHM in OUEM for simplified 3D crown representations and shade, heat and landscape analysis. Single-tree delineation is **not a prerequisite**.

**User-confirmed design constraint:** A model **selected for this project must have a path to commercial use**. This is an eligibility gate, not a metric to trade off against F1. All model comparisons for formal selection must use a provenance-traceable, commercially suitable route, subject to actual license terms and organizational review. Research-only baselines can exist separately but cannot donate pretrained weights, and cannot be treated as licensed production candidates.

Check four distinct layers: (A) implementation and dependencies, (B) initialization/backbone checkpoints and their sources, (C) each training/evaluation image+label and attribution obligations, and (D) resulting weights, distribution and deployment terms. Commercial usability **is not inferred** simply from Apache-2.0 software, a CC BY dataset badge, a generic architecture name, or fine-tuning a restricted checkpoint.

## 2. Evidence ledger

| ID | Finding | Evidence class | Limits |
|---|---|---|---|
| F01 | TCD dataset card describes CC BY 4.0 subset of 4,608 tiles, license-specific NC/SA repositories | Provider claim / public card | Verify actual revision, license per row and source provenance |
| F02 | Official training GT generator uses any non-zero RGB annotation channel for semantic canopy | Source inspection | Not validated against raw images/segments/void/crowd |
| F03 | First-party dataset test asserts distinct source `oam_id` across folds and holdout and one license per split | Source inspection | Current dataset tests not executed |
| F04 | Upstream training code may alter all-empty/all-positive GT at pixel (0,0), and configuration can expose test-as-validation fallback | Source inspection | Exact executed path must be pinned and independently tested |
| F05 | Published Restor pretrained weights have noncommercial/research-linked restrictions; exact SegFormer license references conflict across first-party surfaces | Publisher documentation | No cleared checkpoint established; require artifact-specific license review |
| F06 | Source tiled prediction may resample to ~10cm, crop overlapping tiles, quantize confidence, and potentially blur background/nodata distinction | Source inspection | Not demonstrated as measured production failures |
| F07 | GSI ~0.6m images exhibit variable apparent response to 4x/6x upscaling in prior local PoC | User PoC observation | Not a quantitative domain adaptation benchmark; resizing adds no native detail |
| F08 | Sample viewer `lat`/`lon` appear opposite to coordinates derived from EPSG:3395 bounds | Preliminary record-level metadata comparison | Not universal; require independent bounded sample |
| F09 | Direct HF binary access blocked in current inspection environment | Tool/access outcome | Not a claim dataset is unavailable elsewhere; no pixel QA completed |

For exact provenance, limitations and source URLs see individual AUDIT_01..06.

## 3. Decision register

| Decision | Position | Reason | Reopen condition |
|---|---|---|---|
| D01 Commercial usability | **Required / fixed** | User-confirmed project requirement | Explicit change in project requirements only |
| D02 Reuse Restor released pretrained weights in **commercial-candidate** training | **No at present** | NC and NVIDIA-restricted published-weight provenance | Explicit rights grant or newly verified sufficiently permissive artifact |
| D03 SegFormer architecture | **Open** | Separate architecture from implementation and initial weights | Verified independent implementation/weights plus tests |
| D04 U-Net/ResNet34 architecture | **Open, plausible early candidate** | Simpler semantic architecture; published Restor weights still excluded | Check framework/backbone weights and matched accuracy |
| D05 Random initialization | **Permitted hypothesis, not prescribed** | Avoids checkpoint restrictions but adds optimization/data burden | Performance and compute cost evaluation |
| D06 CC BY-only OAM-TCD data | **Preferred investigation input, not fully approved** | License partition and human annotations are useful | Per-row rights, attribution and GT checks |
| D07 Fine-tuning 10cm→50cm | **Deferred as fixed plan** | Requires a cleared initialization checkpoint and validated downsampled GT | Clean training path + bounded data QA |
| D08 OUEM/model repository interface | **Provisional boundary only** | Geographic processing and model predictions need independent tests | Proven output contract, spatial and tiling tests |
| D09 Use restricted models for comparison | **Separately labeled research-only benchmark, optional** | May provide informative upper/reference baseline | Verify applicable permission, no checkpoint transfer into clean path |

This register is a research decision framework; do not interpret 'preferred' as formal acceptance.

## 4. Priority-ordered issues and acceptance evidence

### P0 — Commercial-clean construction route (blocking formal model selection)
- **I01:** Trace license and provenance of implementation, **all** dependencies, source/repo versions, pretrained encoders or other initial weights, model-card metadata, and resulting artifacts. Document redistribution, attribution and commercial-use conditions.
- **I02:** Verify whether author has released newly cleared CC BY models since original README/paper; resolve contradictory SegFormer license descriptions at artifact level. If ambiguity remains, do not count it as eligible without permission or review.
- **I03:** Identify at least one complete reproducible alternative with explicit rights at every stage (e.g. independently instantiated U-Net with cleared encoder or random initialization). Use a source/provenance manifest, not just a list of library names.
- **Exit gate G0:** At least one candidate has evidence/links, versions, obligations and unanswered legal issues explicitly dispositioned. **No numerical performance requirement at G0.** Legal/compliance signoff when needed.

### P1 — Data feasibility, labeling and evaluation (blocking substantive training)
- **I04:** Pin OAM-TCD CC BY dataset revision, exact train counts, per-row license/source attribution, and ensure no NC/SA contamination. Verify `oam_id` grouping and holdout separation.
- **I05:** Retrieve a bounded **train-only** sample with source RGB + annotation bytes; decode panoptic segment IDs, category schema, background, crowd/void, and exact image-mask alignment, and preserve original data/hash.
- **I06:** Audit `lat`/`lon` versus EPSG:3395 `bounds` in multiple records; don't silently swap on the basis of three viewer rows.
- **Exit gate G1:** Sample-level mask QA documented, reproducible, exceptions recorded; official holdout remains untouched for development tuning. Split/license metadata checks PASS or exceptions dispositioned.

### P2 — Scientific validation of 50–60cm and Japan domain shift
- **I07:** Once G0/G1 pass, compare original native/controlled synthetic low-GSD inputs using a physically described resampling policy and boundary pixel fraction handling. Keep 0.5 GT area fraction and 0.2 source score threshold distinct, and calibrate decision threshold only on validation.
- **I08:** Evaluate eligible models under matched grouped folds on IoU/F1/recall/precision, area bias, spatial errors and quality subgroups (size, biome, boundary). Keep official test separate.
- **I09:** Use GSI Japan imagery to check domain shift; initially label quality-limited qualitative findings as such, secure independent Japan GT for numeric evaluation before nationwide performance claims.
- **Exit gate G2:** Quantitative clean-model performance and stability at ~0.5–0.6m documented. Define performance threshold based on downstream OUEM error tolerance, **not arbitrary invented IoU target**.

### P3 — OUEM contract and deployment
- **I10:** Determine whether model repo emits per-pixel float probabilities and input validity; OUEM manages CRS, geospatial tiling, nodata, mosaicking, vectorization and CHM/3D.
- **I11:** Test resolution assertions, edge padding, stitching, tile seams, spatial alignment, all-background/bright imagery, no-data masks, and calibration; benchmark local CPU and feasible GPU runtimes.
- **Exit gate G3:** Golden raster tests and reproducible run metadata pass; platform integration has an explicit spatial interface contract.

## 5. Recommended sequencing (no automatic implementation)

**Now:** Prepare G0 evidence table for at least two architecture routes, and verify candidate libraries plus backbone initialization rights. Source review can proceed immediately; no need to download full dataset.

**Next:** Pin a small CC BY train-only sample and complete G1 metadata/pixel QA. These checks can occur in parallel with P0 research, but no full training yet.

**Then:** Compare commercially eligible candidates, including controlled synthetic 50cm imagery, before selecting one. Restricted-weight benchmark can be kept separate **if its intended use is permitted**.

**Finally:** Lock OUEM spatial contract and broader evaluation after demonstrable data and model behavior.

## 6. Explicitly out of scope and anti-goals
- Not adopting SegFormer or U-Net prematurely; not claiming a license-cleared production model.
- Not treating CC BY dataset availability as proof of permitted use for an NC-trained derivative.
- Not copying NVIDIA-restricted code or trained weights into a claimed independent route.
- Not equating viewer thumbnails/source tests with full raw-pixel QA or test PASS.
- Not building complete nationwide tiling/3D deployment before model validity is established.
- Not modifying existing experimental plans or merging this Draft PR without review.

## 7. Next research deliverables, in order
1. **Commercial Route Matrix**: U-Net+ResNet34 and SegFormer/MiT-B0 (plus an optional clean third alternative), with exact implementation licenses, initial checkpoints, model card/revisions, data lineage, attribution/redistribution obligations, and explicit green/yellow/red/unknown evidence assessments.
2. **CC BY Row QA**: tiny bounded training-only raw records, source-side metadata/segment decoder checks; exact revision and reproducible output.
3. **Selection Memo**: compare only viable clean routes plus cost, GSD/domain-shift risks; set pass/fail criteria before training.

**Approval boundary:** User-approved hard requirement D01 only. The rest is proposed prioritization and decision gates, not a decision to execute a particular training experiment.
