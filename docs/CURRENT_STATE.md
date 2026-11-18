# OUEM Canopy Segmentation — Current State

As of **2026-11-18**, this is the Phase C closure and experiment-preparation checkpoint for [Issue #82](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/82). Documentation records evidence and decision gates. Execution and acceptance require their own approvals.

## 2026-11-18 current checkpoint — Phase C complete

The Phase C1–C8 research-data engineering and the bounded scientific inventory are complete. Issue #81 was independently reviewed PASS for the inventory scope and closed. Targeted human image confirmation and a new v5 GPU experiment remain pending. The local evidence summarized here is **CODEX_LOCAL_REPORT**. The independent review checked the published report, repository and official documentation without independently opening or hashing private pixels. Evidence classes are defined in the [2026-11-18 Project Review](PROJECT_REVIEW_20261118.md). [Scientific review disposition](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971582740)

| Track | Current evidence / status |
| --- | --- |
| Historical Satlas GPU execution | G1-B T4 step, G2 Train7 / 20 steps with Step10 continuation and Step20 restore, G3-A Val3 inference demonstrated |
| Useful canopy model quality | **NOT MET**. Historical Step20 argmax predicted all background, Tree IoU/F1 zero |
| Fixed 150-image technical QA | PASS 55 / HOLD 94 / FAIL 1 |
| Formal R1 | **0/150 accepted, all 150 HOLD** |
| Phase B geometry pool | Train 91 / Val 19, totaling 110 potentially placeable original images |
| Research v5 | **Train 70 originals / 139 windows + Val 18 originals / 18 fixed windows**, totaling 88 originals / 157 windows / 471 TIFFs |
| v5 research review | `FROZEN_RESEARCH_CANDIDATE_PENDING_INDEPENDENT_REVIEW`. CLEAR records AI screening, with human image confirmation pending |
| v5 training | **GPU run unperformed**. Existing reader/sampler CPU validation is reported, a v5 CUDA entrypoint remains to be implemented |
| Japan/GSI and commercial use | Suitability **NOT VERIFIED**, commercial/production use unapproved |

### v5 identity and scientific limits

These pins are transcribed from the completed C8/scientific reports. This documentation task performed no fresh local artifact hash validation. Locate the external audit root on GEEKOM and reconcile it before reuse. The reported directory identifier is `phase-c8-expansion-20261116-01/dataset-v5`. [Phase C8 report](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/80)

| Reference | Reported SHA-256 |
| --- | --- |
| `dataset-v5/crop_plan.json` | `b6711c33eafed51dd8d42040abf13e3a52601d5be54b08479217ac7e043c8692` |
| `research_review.json` | `ccda53e9dac2c40fda6be718b9fc8a57dc35e6210fcc271a5a5533e390456dbc` |
| `dataset_freeze_v5.json` | `d7a72bdf1c792678cee0030c72dded18d3cdd6639a800c12ebafc63ba3fa1978` |
| Complete dataset identity digest | `a645e831c4214cab913506e01fe9160f5272cf8aed1a6b00bbbf35964cd3b6d7` |

- Individual tree and tree-canopy positives are intended to be merged. The reported eight-source RGB-nonzero / Pillow-L-positive comparison differed by zero pixels. Full segment/category correspondence remains unverified and is not a prerequisite for the bounded exploratory experiment. Annotation-quality questions about shrubs, crops and nursery crowns remain for targeted review. [Official-policy clarification](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6916261686)
- Sixty-nine Train sources have two windows, with median pair IoU 0.590062, 18 pairs above 0.8 and 6 above 0.9. Their union across 139 windows is about 89.68 single-window-area equivalents. This quantity measures geometric area. Statistical effective sample size remains unverified.
- Val18 has 15 positive and 3 all-background windows. Japanese Val461 is a background shoreline window, leaving Japanese positive-tree validation absent. Keep Val18 IDs, locations and products immutable.
- The 110-source pool partitions into 88 research candidates, 16 fully held cases and 6 unexamined sources, IDs 544, 965, 2299, 3062, 3100, 3195. Source614 additionally has one held window. Their processing requires a later scoped decision.
- Source-byte and individual CC BY 4.0/STAC checks are Codex-reported documentary evidence. Physical all-pixel validity, independent human imagery sign-off and final commercial rights remain unresolved. The research route has RGB/fraction/mask products. The [Formal R1a contract](implementation/R1A_DATASET_CONTRACT.md) additionally requires validity fraction and source-bound formal review objects.

These scientific quantities and limitations come from the [Issue #81 report](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6918370260). Earlier v1–v4 hashes, source lineage, held windows and the G0–G3 experiments remain preserved. The [C4 v1 / Val7 procedure](implementation/PHASE_C4_RESEARCH_DATASET.md) retains its historical scope.

### Immediate handoff

On GEEKOM, perform the read-only local evidence/SHA preflight and human review of the pinned 24 CLEAR + 9 held contact-sheet windows. Prioritize the CLEAR/held nursery and planting-row contrasts, linear teacher edges and known coverage margin. Then prepare a bounded research experiment plan with fixed Val18 and a small reviewed CUDA bridge. Follow the gates in the [current Roadmap](ROADMAP_NEXT.md).

`ResearchData` and `r2_train.make_plan` provide the research input/source-balanced schedule. `r2_train.train` enforces CPU FP32. The historical G2 CUDA entrypoint is fixed to Zenodo10 / 20 steps, and G3-A uses historical Val3. A new v5 GPU integration and its execution remain pending. Confirm CUDA capacity on the chosen T4 or other runtime. GEEKOM's role as the image-review host does not establish CUDA availability.

The source history is [Phase B #68](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/68), [C4 #75](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/75), [C5 #77](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/77), [C6 #78](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/78), [C7 #79](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/79), [C8 #80](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/80), and [scientific review #81](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81).

## Historical checkpoint — 2026-11-14

The following checkpoint and the 2026-11-07 sections preserve their dated content. Documentation PRs #41 and #65 were merged. Earlier draft/proposal wording records its decision-stage provenance, while run-specific approvals remain separate.

Original as-of statement was 2026-11-14 policy/status proposal, preserving 2026-11-07 evidence. Original status was Draft for review, with technical pipeline PASS and useful model/M1 completion not yet established.

## 2026-11-14 checkpoint (newer than the 2026-11-07 reference below)

- **Scientific outcome:** The Satlas-first backbone/FPN + two-class SEGMENT route remains preferred. The old T4 20-step training and restore/inference pipeline worked; its three fixed Val crops gave Tree IoU/F1 = 0 with all-background predictions. M1 model quality and GSI suitability remain **NOT MET / NOT VERIFIED**, respectively.
- **Data:** Fixed OAM-TCD candidate pool 150 (Train 120 / Val 30) has Technical PASS **55** (Train 49 / Val 6), HOLD **94**, FAIL **1** (4508 grayscale TIFF). **Formal R1 accepted = 0**; Technical PASS is not training admission. Existing Zenodo10 are smoke-only.
- **Pilot:** Train 717/1054 and Val 1241/461 are fixed for the R1d four-image workflow. PR [#61](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/61) added bounded synthetic preparation, and PR [#63](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/63) merged G3/G4 proposals plus evidence-only readiness (main `c8e6eac2a6b6e20dc166c15d8f2e9d4b8573b538`). The four real GeoTIFF sets were **NOT generated**. G3/G4 and human execution approval are **PENDING**, all formal candidates remain HOLD, and the existing real-mode gate is `REAL_EXECUTION_DISABLED`.
- **Sandbox:** User-observed Windows Sandbox feature enabled, disabled-network guest startup succeeded; no guest Python/GDAL/Job/output smoke has been run. Issue [#64](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/64) remains open but can be **paused**, as an optional environment experiment rather than a prerequisite for ordinary coding, synthetic tests, data planning or Trainer development.
- **Proposed risk-proportionate policy:** [Design Principles](DESIGN_PRINCIPLES.md#risk-proportionate-execution-policy-2026-11-14-proposal) and [Next Roadmap](ROADMAP_NEXT.md#2026-11-14-priority-reset--focus-on-useful-training-not-compulsory-isolation). Focus on data supply, a generalized Trainer and meaningful model assessment. Sandbox and mandatory zero-byte network proofs should not be blanket requirements; any real-data execution route remains subject to a separate explicitly approved, source-bound contract and applicable code-gate changes. No license, G3/G4, `ALL_VALID`, R1 or real execution permission is conferred by this proposed note.

**Historical reference:** Sections 1–6 below retain the 2026-11-07 evidence and pinned artifacts. They are not silently rewritten into the 2026-11-14 execution outcome.

## 1. Active direction

- **Target:** GSI aerial RGB (~0.5–0.6 m/local-ground pixel) to reproducible 2D Tree/Canopy probability and mask; subsequent geospatial/3D/thermal/landscape work belongs to OUEM.
- **Preferred model:** Ai2 SatlasPretrain `Aerial_SwinB_SI` official Swin-B backbone/FPN plus freshly initialized official 2-class SEGMENT head, fine-tuned on license-verified OAM-TCD data.
- **Source generation:** Zenodo `11617167-v1.0.0` is the active source for the existing smoke set; the previous Hugging Face inventory and black-border pilots remain historical evidence. Avoid silently mixing their metadata or license assertions.
- **Current processing:** near-0.50m local-ground scale transformation, aligned RGB/fraction/binary/validity rasters; fixed labels, thresholds and provenance. Do not claim native GSI-equivalent image fidelity.
- **No public release or redistribution of trained model/checkpoint weights is planned**. Internal and potentially commercial inference remain intended uses, subject to a separate final organizational rights review.

## 2. Verified small-run lineage

| Step | Scope and evidence | Status |
| --- | --- | --- |
| Windows CPU model/load/real-data preparation | M1-1 to M1-4B bounded experiments, CPU update and continuation tests | PASS within recorded conditions |
| Data inventory and expansion pilots | M1-5A / M1-5B / M1-5C / M1-5D: 4,608-record inventory, black-border HOLD investigation, canonical Zenodo local-adapter tests and bounded candidate selection | Partly accepted; HOLD samples excluded |
| G0 | Frozen Zenodo10, Train7/Val3, 40 TIFFs + manifest, exact SHA-bound transfer | PASS for smoke only |
| G1-A | Dataset ZIP staging, Drive copy and SHA checks | PASS |
| G1-B | Tesla T4 one-step optimizer execution and state restore | PASS |
| G2 | Train7 bounded 20-step training, Step10 continuation, Step20 checkpoint and restore evidence | PASS for pipeline, not performance |
| G3-A | Two-model same-Val3 GPU evaluation, metrics, PNGs, artifacts backed up and SHA verified | PASS for execution; segmentation quality NOT MET |
| G3-D | Additional raw-probability / rank / calibration diagnostics | DEFERRED |
| GSI qualitative model evaluation | New Satlas fine-tuned model on several real GSI scenes | NOT RUN / NOT ACCEPTED |
| Generic large-data training and OUEM handoff | Variable-size data/Trainer/general inference package | NOT IMPLEMENTED / NOT ACCEPTED |

**Acceptance boundary:** The ten archived G0 sample records are accepted for **smoke testing only** and each records `category_level_per_id_validation=NOT_VERIFIED`. R1 may test these as backward-compatible fixtures, but must make separate, auditable QA dispositions before counting them toward its new training-dataset target.

## 3. Pinned smoke references

| Asset | Identifier |
| --- | --- |
| Zenodo10 dataset ID | `ouem-zenodo10-e3db510b1125931d` |
| Zenodo10 manifest SHA-256 | `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca` |
| Frozen Zenodo10 transfer ZIP SHA-256 | `db282971016895a23052c3006992d3a06c5008c9d98b4e23e1551c1d16ab8521` |
| Ai2 initial model SHA-256 | `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036` |
| G2 Step20 checkpoint SHA-256 | `47044e414c27718d37ea2d67a89f88cc24e2777757db07cfad2286ead2b9e767` |
| G3-A run ID | `m1-7-g3a-01` |
| G3-A commit | `0d5942bcc86f70c3d62ee01f6ef13ec216fb0116` |

Google Drive record locations (not stored in Git; exact SHA and authorizations must be rechecked before re-use):

- Root: `MyDrive/OUEM_Canopy_Segmentation/`
- Data: `02_datasets/zenodo10_g0/zenodo10-g0.zip`
- Official weights: `01_pretrained/aerial_swinb_si.pth`
- Step20: `03_experiments/m1-7-g2-01/step20/zenodo10_g2_step20.pth`
- G3-A evaluation: `03_experiments/m1-7-g3a-01/` (`metrics.json`, `provenance.json`, `result.json`, `sha256.json`, three `val_*.png`)

**Evidence precision:** G3-A reported `G3A_EVALUATION_AND_COPY_PASS`; the present review verified Drive file presence and receipts, but did not independently rehash the entire >1 GB G2 weight artifact. Distinguish stored run-level hash evidence from review-time byte verification.

## 4. G3-A performance evidence

Same three source-domain Val top-left 224×224 crops: IDs **1972, 269, 3785**, 50,176 pixels each.

| Metric (micro unless noted) | Reconstructed Step0 | G2 Step20 |
| --- | ---: | ---: |
| Tree IoU | 0.342478 | 0 |
| Tree F1 | 0.510218 | 0 |
| Precision | 0.342691 | undefined (no predicted positives) |
| Recall | 0.998187 | 0 |
| Predicted canopy fraction | 0.992679 | 0 |
| Pixel-mean cross-entropy | 0.696020 | 0.611718 |

Step0 Head/Upsample equality with the original training-time initialization is **NOT_VERIFIED**. At Step20, binary prediction is all background; lower CE alone is not an accepted segmentation improvement. Exploratory PNG confidence differences between GT-tree and GT-background are not proof of usable boundaries, nor a basis to optimize a validation threshold. The three crops are **not whole-image validation** and do not establish GSI performance.

## 5. Important open items

- **Data:** first expansion to **~100 R1-accepted images** (provisional 80 Train / 20 Val; archived Zenodo10 images count only after R1 requalification); rights, mask categories, black-border validity, source scene independence and meaningful geographic/tree-cover diversity; no holdout/GT54 leakage.
- **Model:** yet to deliver demonstrably useful canopy masks; loss behavior on larger data, source-domain generalization, and GSI visual suitability unknown.
- **Scaling:** hard-coded Zenodo10 sample/size constraints, 20-step-only Trainer, per-step full dataset validation, fixed Val3 crops, Colab interruption resilience, compute cost.
- **Commercial:** source attribution and Satlas/ImageNet lineage reviewed at an appropriate deployment gate; no current commercial production signoff.
- **OUEM:** fixed inference/tiling/validity/GSD/threshold/geospatial contract and downstream 3D integration not yet accepted.

## 6. Project controls

1. Preserve accepted G0–G3 code, datasets, checkpoints, hash receipts and output directories; do not overwrite for new experiments.
2. New work must use distinct dataset IDs, runs and manifests with reproducible seeds/config; distinguish code tests from real GPU evidence.
3. Keep source `oam_id`/fold grouping and geospatial independence; official OAM holdout and GT54 remain protected.
4. Use [Project Charter](PROJECT_CHARTER.md) for purpose, [Project Review](PROJECT_REVIEW_20261107.md) for the review disposition, and [Roadmap](ROADMAP_NEXT.md) for proposed new phases; all documents remain subject to review until merge.
