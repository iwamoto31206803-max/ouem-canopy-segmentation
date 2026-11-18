# OUEM project review — 2026-11-18 Phase C closure

This documentation checkpoint completes [Issue #82](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/82). The scoped scientific inventory in Issue #81 is complete and independently reviewed PASS. Its recommendation for bounded research learning is **CONDITIONAL GO**, with execution approval and targeted human image review pending. Useful canopy performance remains **NOT MET**. [Independent disposition](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971582740), [closure checkpoint](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971780623)

## Purpose and authority

Build a practical RGB 2D canopy component for Japanese GSI imagery at approximately 0.5–0.6 m/ground pixel. OUEM owns downstream CHM/3D, thermal and landscape integration. Preserve Ai2 SatlasPretrain `Aerial_SwinB_SI`, its official backbone/FPN and a freshly initialized two-class SEGMENT head. Source-bound CC BY 4.0 OAM-TCD data support the research route. Commercial deployment and model/checkpoint rights require separate review. [Project Charter](PROJECT_CHARTER.md), [Design Principles](DESIGN_PRINCIPLES.md), [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md)

The charter, principles and ADR retain precedence and original decision-stage provenance. Documentation PRs #41 and #65 were merged. A checked-in document records design/evidence status, while run authorization, human quality judgments and production acceptance remain separate decisions. The [Current State](CURRENT_STATE.md) is the latest factual entry point and the [Roadmap](ROADMAP_NEXT.md) defines the next gates.

## Evidence classes and review limits

| Class | Meaning in this review |
| --- | --- |
| `GITHUB_VERIFIED` | Repository implementation and published issue/review records inspected for this update. Published run claims retain their original evidence scope. |
| `CODEX_LOCAL_REPORT` | External artifact measurements, source-byte/STAC verification and AI image observations reported by the earlier Codex tasks. Private bytes were not re-opened or rehashed during this documentation task. |
| `NOT_VERIFIED` | Evidence remains absent, pending independent image review or pending the future experiment/deployment decision. |

The reviewed baseline is main `c38e41864cc0dca1d735b49315704eda186008cf`, PR #76. C5–C8 expansion and Issue #81 added external artifacts and issue reports without later code commits. This PR changes four documentation files. No local raw data, datasets, weights or audit receipts were accessed for this update.

## Attained milestones and boundaries

| Milestone | Evidence / disposition |
| --- | --- |
| Historical G1-B | Tesla T4 one-step Satlas learning and checkpoint/restore demonstrated in the published run record |
| Historical G2 | Train7 / 20 optimizer steps, Step10 continuation and Step20 state restoration demonstrated |
| Historical G3-A | Val3 inference, quantitative/visual outputs and backup demonstrated |
| Useful model quality | **NOT MET**. G2 Step20 argmax gave all-background output, Tree IoU/F1 zero on three fixed crops |
| Phase C1–C8 | Research patch engineering and staged data expansion complete |
| Issue #81 inventory | Scoped scientific review complete, independently reviewed PASS |
| Targeted human imagery QA | **NOT_VERIFIED**. The private 24 CLEAR + 9 held observations remain AI screening |
| Research v5 CUDA training | **Unperformed**, with a narrow implementation bridge still needed |
| Formal R1 | **0/150 accepted, all 150 HOLD** |
| GSI practical suitability / commercial use | **NOT_VERIFIED / unapproved** |

Historical T4 records are linked through [G1-B PR #38](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/38), [G2 PR #39](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/39), [G3-A PR #40](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/40) and the preserved [2026-11-07 state](CURRENT_STATE.md#2-verified-small-run-lineage). The reconstructed Step0 head was not assured identical to the original initialization. Lower historical CE alone did not establish useful canopy predictions. Those checkpoints and results remain immutable.

## Research dataset lineage and current pins

Phase B found 110 geometrically placeable originals, Train91 / Val19. This is a potential pool. The research snapshots use approximately 0.50m area aggregation and direct 224×224 windows avoiding positive-area support of edge-connected exact-black regions. Black/white colors alone do not establish NoData or physical validity. [Phase B #68](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/68)

| Snapshot | Train originals / windows | Val originals / windows | Source record |
| --- | --- | --- | --- |
| v1, C4 | 13 / 26 | 7 / 7 | [C4 #75](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/75) |
| v2, C5 | 25 / 50 | 18 / 18 | [C5 #77](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/77) |
| v3, C6 | 42 / 84 | 18 / 18 | [C6 #78](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/78) |
| v4, C7 | 59 / 118 | 18 / 18 | [C7 #79](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/79) |
| v5, C8 | **70 / 139** | **18 / 18** | [C8 #80](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/80) |

C1–C3 established the pilot, label/input review and expanded source-balanced supply. Their records remain [C1 #69](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/69), [C2 #71](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/71) and [C3 #73](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/73). The [C4 v1 document](implementation/PHASE_C4_RESEARCH_DATASET.md) preserves its Val7 and 200-step proposal as an archived snapshot. New v5 execution planning uses fixed Val18.

v5 totals **88 originals / 157 windows / 471 TIFFs**, with historical source Technical PASS45 / HOLD43. It retains the artifact status `FROZEN_RESEARCH_CANDIDATE_PENDING_INDEPENDENT_REVIEW`. Completing the report review has not rewritten its private freeze or supplied human image sign-off. Previous v1–v4 pins and Val18 products remain preserved according to the local reports.

The external directory identifier is `phase-c8-expansion-20261116-01/dataset-v5`. Reconcile the actual GEEKOM location and reviewed pins during the next read-only preflight. The following values are `CODEX_LOCAL_REPORT` references transcribed from Issue #80/#81.

| Artifact | SHA-256 |
| --- | --- |
| `dataset-v5/crop_plan.json` | `b6711c33eafed51dd8d42040abf13e3a52601d5be54b08479217ac7e043c8692` |
| `research_review.json` | `ccda53e9dac2c40fda6be718b9fc8a57dc35e6210fcc271a5a5533e390456dbc` |
| `dataset_freeze_v5.json` | `d7a72bdf1c792678cee0030c72dded18d3cdd6639a800c12ebafc63ba3fa1978` |
| Complete dataset identity digest | `a645e831c4214cab913506e01fe9160f5272cf8aed1a6b00bbbf35964cd3b6d7` |
| Scientific `scientific_review.json` | `9ced56d27a52a0741e2f0447e73b5be6d57bb344c060526124d61cdd8102d55d` |

The 110-source pool partitions into 88 research candidates, 16 fully held cases and 6 unexamined IDs, 544, 965, 2299, 3062, 3100, 3195. Source614 also has a held window alongside its retained window. Further processing requires a scoped decision. About 100 originals remains an aspiration and carries no acceptance floor.

## Scientific conclusions and residual risks

### Teacher meaning and quality

Official OAM-TCD supports binarizing Individual tree and Tree canopy together. The reported eight-source comparison found zero pixel differences between nonzero RGB and Pillow-L positivity. The local COCO mapping is 1=canopy, 2=tree. COCO annotation IDs remain distinct from panoptic segment IDs. The representative check supports the current union policy, while a full 88-source crosswalk remains unverified. [Official dataset policy](https://restor-foundation.github.io/tcd/dataset/), [official semantic reader](https://restor-foundation.github.io/tcd/reference/tcd_pipeline/data/cocodataset/), [Issue #81 clarification](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6916261686)

The substantive quality questions concern omissions, low vegetation/crops/nursery crowns, ambiguous linear positives and artificial margins. Nursery or palm content alone does not prove a wrong annotation. The 24 CLEAR + 9 held purposive AI observations cannot estimate a dataset error rate. Prioritize human comparison of 3380/4066, 802/727, 2006/614/4139 and contrast cases 2991/4315. Polygon spatial probes were approximate and some RLE/PNG differences remain documented. Preserve labels and uniform masks. [Authors' tree definition](https://restor-foundation.github.io/tcd/introduction/#what-is-a-tree), [scientific inventory](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6918370260)

### Diversity and Val18

Sixty-nine Train originals have two windows and 614 has one. Pair IoU median is 0.590062, with 18 pairs above 0.8 and 6 above 0.9. The geometric union of 139 windows is about 89.68 single-window-area equivalents. Independent-scene count and statistical effective sample size remain unverified. Nearby distinct source assets can share acquisition context. Source-balanced sampling controls source weight while these correlations persist.

Reported Train window teacher mean is 24.862%, median 12.911%, and source-equal mean 24.777%. Val mean is 19.355%, with 15 positive / 3 all-background windows. Japanese Val461 is a background shoreline crop, so positive Japanese urban-tree validation is absent. Fixed Val18 can diagnose within-version OAM behavior. Source-domain generalization and GSI accuracy claims remain unsupported. Preserve all Val18 IDs, crop coordinates and products. [Independent scientific review](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971582740)

OAM imagery area-resampled to approximately 0.50m can differ from native GSI optics and seasons. Small crowns may disappear under the saved-fraction 0.5 threshold. A later predefined GSI qualitative review remains the next practical target-domain gate. The charter does not require mass Japanese ground-truth collection for this first suitability decision.

### Rights, formal acceptance and engineering readiness

The source-specific CC BY 4.0/STAC and source/product SHA checks are `CODEX_LOCAL_REPORT`. The documentation independent review did not rehash those private files. Final commercial/model rights and all-pixel coverage claims remain unresolved. The fixed 150-image technical QA remains PASS55 / HOLD94 / FAIL1. All 150 Formal R1 outcomes remain HOLD.

Research windows contain `rgb.tif`, `canopy_fraction.tif` and `canopy_mask.tif`. The [R1a contract](implementation/R1A_DATASET_CONTRACT.md) separately requires four products including `valid_fraction.tif`, together with source-bound rights/category/validity/independence/R1 reviews. Research CLEAR provides no formal acceptance assertion.

Repository inspection is `GITHUB_VERIFIED` for the implementation boundary. [ResearchData](../scripts/research_data.py) and [make_plan](../scripts/r2_train.py) support the pinned research inputs and source-balanced schedule, with CPU validation reported. `r2_train.train` enforces CPU FP32. [Historical G2 CUDA](../scripts/m1_7_g2_train.py) remains bound to Zenodo10 / 20 steps. Fresh v5 CUDA integration is pending and v5 GPU learning is unperformed. GEEKOM/Radeon host access supports local review without demonstrating CUDA availability.

## Next decisions and deferred work

1. Reconcile local v5/evidence/SHA and obtain targeted human judgments on the private 33-window review set. Record uncertainty and follow-up without rewriting frozen v5.
2. Review a bounded exploratory experiment plan with input/initializer identities, fixed Val18, source-balanced sampling, metrics, environment, new outputs, resource/step budget, stop policy and checkpoint backup/restore.
3. Review and merge a small research CUDA bridge reusing G1-B/G2/G3-A primitives. Then authorize a smoke/main run separately when a measured CUDA runtime is available.
4. Assess source-domain confusion counts, macro/micro IoU/F1, precision/recall, CE and area bias with undefined metrics recorded explicitly. Proceed to predefined GSI qualitative scenes only from observed model evidence.

The [Roadmap](ROADMAP_NEXT.md) specifies these gates. Further source expansion, final-six screening, held-image rescue, valid-mask/loss changes and threshold optimization remain deferred. Administrative Issues #50/#56 and paused Sandbox #64 require separate triage. No execution or issue closure is implied by this document.

The prior [2026-11-07 review](PROJECT_REVIEW_20261107.md), [2026-11-11 review](PROJECT_REVIEW_20261111.md), dated Current State/Roadmap sections, ADR and experiment evidence remain historical records. This handoff preserves their design lineage and distinguishes current research readiness, unmet model quality and pending formal/production decisions.
