# OUEM Canopy Segmentation — Project Review (2026-11-07)

**Status:** Review draft. Documentation-only; subject to review/merge. No retraining, new model acceptance, rights clearance, or production approval is implied.

## 1. Purpose and scope

The purpose remains a maintainable, reproducible, and potentially commercially usable **2D RGB tree-canopy semantic-segmentation component**, intended for Japanese GSI aerial imagery around **0.5–0.6 m/local ground pixel**. OUEM—not this repository—owns georeferencing/large-area imagery, vectorization, point-cloud/CHM-derived 3D canopy, and shade/thermal/landscape evaluation.

The output mask is an intermediate layer. Do not mistake model-only IoU for OUEM-level effectiveness. Single-tree/instance segmentation, exhaustive architecture comparisons, a research paper, mandatory new Japanese GT for M1/M2, and nationwide accuracy certification are outside the immediate milestone.

**Source hierarchy:** [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md), and [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md) remain authoritative. This review records decisions and evidence; it does not silently supersede those documents.

## 2. Four-part review — findings

| Area | Finding | Decision |
| --- | --- | --- |
| Mission and boundaries | The 2D canopy ↔ OUEM 3D/geospatial responsibility split remains appropriate. | **KEEP** |
| Architecture | Official Ai2 SatlasPretrain `Aerial_SwinB_SI` backbone/FPN plus new two-class SEGMENT head runs in a real GPU workflow. Practical model quality is not established. | **KEEP** as initial candidate; no architecture search now |
| Dataset and labeling | A near-0.50m ground-scale transformation, hash-bound manifests, grouped splits and source QA are implemented for small, accepted inputs. Source rights, missing validity, and semantic uncertainty continue to limit broad expansion. | **EXPAND** with evidence-based QA |
| Engineering | A fixed Zenodo10 training/evaluation pipeline with checkpoint, restore and visualization works. Hard-coded IDs/steps/regions and repeated data validation do not constitute a scalable Trainer. | **GENERALIZE** in separate new code; preserve G0–G3 |
| Model quality | On three **fixed 224×224 Val crops**, G2 step20 predicts all background at the untuned argmax rule; Tree IoU and F1 are both 0. | **NOT ACCEPTED** for useful segmentation |
| Operations/rights | GPU runs and backup artifacts exist. Actual Japan/GSI suitability, continuous training affordability, downstream integration and commercial clearance are still open. | **DEFER/VERIFY** in the relevant later gates |

## 3. Evidence-backed technical milestone

**Engineering pipeline: achieved for a bounded small-data experiment (not M1 model-quality completion).**

- Existing Zenodo10: 7 Train + 3 Val, exact manifest `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca`; 40 derived GeoTIFF products. Its acceptance scope is **`accepted_for_M1_7_smoke`**, not general training-quality certification. All ten sample records retain `category_level_per_id_validation=NOT_VERIFIED`; R1 must separately assess training-dataset eligibility before counting any of these ten as accepted for expanded training. Do not modify the frozen G0 manifest or prior results.
- G1-B: real Tesla T4 CUDA optimizer step and checkpoint-state restoration.
- G2: fixed 20 optimizer steps, Step10 checkpoint continuation and Step20 checkpoint. Step20 SHA-256: `47044e414c27718d37ea2d67a89f88cc24e2777757db07cfad2286ead2b9e767`.
- G3-A: real GPU evaluation and Drive copy PASS with paired JSON metrics/provenance and three PNG visualizations. Reconstructed Step0 **is not proven identical** to original G2 initial head/upsample; only the official backbone/FPN identity is anchored.
- G3-A micro results: reconstructed Step0 Tree IoU **0.342478**, F1 **0.510218**, CE **0.696020**; Step20 Tree IoU **0**, F1 **0**, CE **0.611718**. Reconstructed Step0 predicts ~99.27% canopy, so its nonzero IoU does **not** demonstrate a useful segmentation baseline. Step20 predicts 0% canopy. These are three fixed source-domain crops, not GSI accuracy.
- Exploratory PNG-based class-probability statistics suggested tree/background separation; they are approximate, not raw-logit AUC/AP or formal new acceptance metrics. Detailed G3-D diagnostics are deferred.

Runtime receipts are caller-run GPU observations and Google Drive artifacts; repository stage documents may predate those executions and can retain `NOT_VERIFIED` statements about **their own original publication time**.

## 4. Priority risks and constraints

1. **Dataset QA:** Keep item-level CC BY 4.0 rights, source attribution, category/label semantics, valid-area evidence, exact RGB/mask alignment, and `oam_id` + geospatial separation. Historical black-border HOLDs must not be turned into background/NoData by color alone. The active Zenodo data lineage must not inherit conflicting/missing rights fields from the older Hugging Face metadata view.
2. **Generalization:** Source-region/sensor/season changes and GSI domain shift are untested. At ~0.50m output GSD, reducing 0.1m source images is not identical to native lower-resolution GSI capture.
3. **Compute:** T4 bounded training works; multi-epoch runtime, memory, Google Colab session interruption handling and throughput remain unmeasured at ~100-image scale.
4. **Training/evaluation architecture:** `m1_7_dataset.py` requires exactly Zenodo10; G2 schedules 20 steps with fixed checkpoint gates; G3-A evaluates three fixed crops. Preserve these as audit fixtures; build separate generalized components.
5. **Rights:** Ai2's weight designation, code license and OAM-TCD CC BY evidence are positive inputs, **not** production commercial clearance. Upstream ImageNet weight lineage remains a separately recorded legal/organizational question.
6. **State reporting:** README needs an as-of update, distinguishing recorded GPU execution from older implementation-era notes.

## 5. Review disposition and next work

**Review disposition: CONDITIONAL PASS.** No fundamental mission or architecture rebase is currently justified. Shift effort from repeated bounded smoke tests toward a clean ~100-image training dataset, general training/restore, then early qualitative GSI inspection. Do **not** interpret a dataset-size target as permission to lower data QA.

This review records the user's revised first expansion target: **approximately 100 total accepted images**, provisionally **Train 80 / Val 20**, with the official validation folds and source-group separation taking precedence over the ratio. The target aims to include the existing ten **only where they pass the R1 acceptance rules**; smoke acceptance alone does not promote them. Actual accepted counts are evidence-driven. The second expansion scale is **not committed**.

See [Current State](CURRENT_STATE.md) and [Next Roadmap](ROADMAP_NEXT.md). No code/checkpoint/data artifact is changed by this review PR.
