# OUEM Canopy Segmentation — Current State

**As of:** 2026-11-14 policy/status proposal; preserved 2026-11-07 evidence below  
**Status:** Draft for review. **Technical pipeline PASS / useful model and M1 completion NOT YET ESTABLISHED**.

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
