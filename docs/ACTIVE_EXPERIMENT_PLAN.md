# Active Experiment Plan — Zenodo-first canopy fine-tuning (M1-7)

**Status: PROPOSED design for review (2026-11-06). No new training, model accuracy, acceptance or license sign-off is claimed.**

**Authority:** [Project Charter](PROJECT_CHARTER.md) > [Design Principles](DESIGN_PRINCIPLES.md) > [Satlas-first ADR](decisions/ADR_20261101_SATLAS_FIRST.md). This is the **active execution plan** anticipated by the charter; [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) is a historical, superseded Restor-checkpoint A/B/C study, **not** the current execution instruction. Detailed Zenodo decisions/evidence remain in [M1-5A inventory and M1-5D notes](implementation/M1_5A_DATASET_INVENTORY.md) and [M1-5B pilot/result ledger](implementation/M1_5B_PILOT_RESULT.json).

## 1. Purpose and non-goals

**Primary practical question:** Can the licensed Ai2 SatlasPretrain \`Aerial_SwinB_SI\` initializer, with the official FPN and fresh two-class semantic head, be fine-tuned reproducibly on verified Zenodo OAM-TCD imagery at **~0.50 m actual ground GSD**, then produce usable tree-canopy predictions suitable for initial visual inspection on **GSI ~0.5–0.6 m imagery**?

The next immediate deliverable is **a working, source-audited multi-image GPU training/validation/checkpoint/inference pipeline**, not a scientific proof of nationwide GSI accuracy or optimal model selection. The fine-tuning pipeline, rights, engineering fitness and qualitative GSI behavior are each distinct decision gates. Do not add an obligatory scratch comparator, Restor noncommercial weights, a three-arm original/degraded comparison, arbitrary accuracy targets, new Japanese reference labels, or black-border rescue before observing the initial model behavior. No threshold/hyperparameter search on GT54 or OAM official Test/Holdout.

## 2. Dataset lineage and admission

**Prospective training source:** Zenodo record [11617167, v1.0.0](https://zenodo.org/records/11617167), \`source_generation=zenodo-11617167-v1.0.0\`. Read authenticated local RGB TIFF + RGB Panoptic PNG + official JSONL. Embedded Zenodo TIFF CRS/affine governs this distributed source generation; apply the already accepted latitude-aware ~0.50m, pixel-overlap area resampling, binary target at fraction >=0.5, and hash-verified four-product output. Never substitute prior HF original TIFFs/grids/hashes or merge different source generations under one dataset identity.

**Initial *candidates*, prior to formal dataset promotion:**
- Train 7: **3, 661, 1022, 1280, 2243, 493, 1358**
- Val 3: **1972, 269, 3785**

M1-5D-2 generated the first three; M1-5D-3 generated the other seven. They are reported as \`PASS_CANDIDATE\`, source-bound external rights/visual evidence and deterministic product hashes. **Do not infer that they are already accepted into a combined training dataset.** A small manifest acceptance/promotion PR must bind exactly these image IDs, \`oam_id\`, validation_fold, source_generation, per-image original/processed SHA-256, transforms, license/STAC/attribution evidence, per-image review record and distinct source groups, then mark a frozen \`accepted_for_M1_7_smoke\` state. Verify the external on-disk bytes against Git receipts when staging for GPU. Do not invent external evidence or claim to have inspected private images in Git-only review.

Official \`train_meta.json\` folds 1–4 are Train, fold0 is Val; \`test_meta.json\` is official Holdout and **must never be used for training, hyperparameter decisions or initial validation**, even though the files share a directory named \`holdout/\`. Keep image- and \`oam_id\`-group and cross-role footprint checks; preserve held IDs (including **4235** and eleven M1-5D-3 HOLD images). Only explicit per-row CC BY 4.0 with checked source STAC and attribution is eligible. Unknown/null, NC/SA records remain held. Commercial final licensing sign-off is separate.

**Labels:** two-class target \`0=visible background, 1=Tree∪Canopy\` using pinned upstream \`np.any(panoptic_rgb != 0, axis=2)\`. Keep source Panoptic untouched. Per-Panoptic-ID category mapping remains NOT_VERIFIED for new Zenodo images; this limited official binary semantics is NOT proof of perfect label accuracy. \`valid_fraction==1\` is mandatory for the initial clean-only set; no black-to-NoData, crop rescue or unvalidated validity masks. Preserve uniform labels exactly. The accepted 10 are a **functional tiny set**, not a statistically representative train/Val benchmark.

## 3. Execution stages and evidence gates

### G0 — Frozen Zenodo smoke dataset, *separate small PR*

Create a **single combined Zenodo-only manifest** from the existing 10 candidates, retaining their separate original run manifests/immutable SHA receipts. Read and verify the 40 expected products (4 per image) under their existing external directories; create a *fresh, separate* consolidated staging area only if necessary, with byte-for-byte copies or hard links verified by SHA. No re-derivation or silent adjustment. Explicit human review/promotion of the candidate set and conflict-free Train/Val split before any training; no requirement to scale data here. Store acceptance evidence in Git without raw images/private STAC contacts.

### G1 — GPU hardware/optimizer preflight, *separate reviewed code/experiment PR*

The existing [M1-6 GPU smoke](../scripts/m1_6_colab_gpu_smoke.py) is **hard-coded to historical HF Train661/Val1972**, a two-sample old manifest, and top-left crops (Train 256, Val 224). The [M1-3A loader](../scripts/m1_3a_loader.py) similarly assumes IDs 661/1972 and one sample per role. **Neither is a correct Zenodo10 loader** and neither should accept a fabricated HF-style manifest. Keep historical M1-6 and its checkpoint/resume evidence unchanged; use its measured GPU/resource/checkpoint mechanism as a reference for a new small Zenodo-specific entry point or refactor by an explicitly tested adapter with *strict generation dispatch*. If using the original two-image HF run purely for CUDA wiring, label it a separate **legacy infrastructure smoke**, never a Zenodo training or validation result.

Colab has previously passed Tesla T4 CUDA, mounted Drive, copied the earlier HF smoke two-image data and hashed the official initializer. **A full Zenodo GPU optimizer step and resource measurement is not yet demonstrated.** Copy the frozen Zenodo manifest/products to the mounted Drive/Colab workspace with input SHA validation (no implicit download). Verify GPU versions, official package \`satlaspretrain-models\`, fixed initializer hash, full model + FPN + two-class head, output/target dimensions, numerical finite CE/gradients and one optimizer update. Begin with a **224×224, batch1, all-valid** patch (the smallest initial Zenodo output dimension is 230px, so 224 fits all ten); validate model stride/shape first. Adjust patch, batch, precision, training freeze policy **only from measured GPU evidence**, using a separate recorded configuration. Do not rely on assumed 14.56GiB capacity alone. A full frozen-parameter AdamW state may change memory use after the first step.

### G2 — Bounded multi-image optimizer smoke, *only after G1*

Train on **all seven approved Train scenes**, not only one endlessly repeated patch. Implement deterministic, hash-verified scene/patch selection with a recorded seed and reproducible sampler; random train patches may be introduced after verifying image/mask/fraction/validity crop alignment. Keep **all three Val scenes out of gradient updates** and avoid fold reassignment. For initial run use the same approved model initialization, CE target semantics, fixed 0.5 decision / argmax, fixed optimizer settings unless an actual G1 blocker requires a documented change. Set a **small bounded step count (initially <=20 optimizer steps)** for plumbing/overfit diagnostics, not a claim of convergence. Log step-wise loss, gradients, VRAM peak allocated/reserved, step duration, per-scene ID/patch sampling counts, machine/package/commit details, strict data source_generation and manifest SHA. The checkpoint lives in a fresh, persistent Drive or external destination, with model/optimizer/RNG state + code/data/config IDs and independent SHA256. Verify restore (and continuation when feasible) against the exact Zenodo dataset signature; never resume an old HF- or different-data-generation checkpoint as though compatible.

### G3 — Diagnostic validation and GSI qualitative check

Before/after metrics on **frozen** Val images/patches: positive-class TP/FP/FN, Tree IoU, F1, precision, recall, CE, canopy area ratio/bias; report **per-scene** figures plus an explicitly named aggregate, not only pooled pixels. Undefined denominators are null / NOT_DEFINED (never silently perfect). Exclude no pixels in initial all-valid set; do not interpret Satlas's \`gt>0\` as a validity mask because in OUEM \`0=real background\`. Fixed, non-training Val region selection must be documented (e.g. deterministic stride-compatible crop, noting excluded borders); later whole-tile evaluation needs its own tiling/padding/ignore-boundary contract. A short 20-step smoke on Train7/Val3 establishes **execution and directional behavior only**, not statistically reliable model accuracy, fine-tuning benefit at 0.50m, or Japan/GSI generalization.

After the source-domain smoke and saved/reloaded checkpoint are functioning, perform **qualitative GSI inference** on a few previously used environments (e.g. Hanamigawa, Aobanomori and contrasting urban/shaded areas), respecting the actual GSI scale/grid. Inspect missed crowns, shadow/building false positives, fragmentation, canopy-area plausibility and inference seams. Tree-fraction alone is not accuracy. Keep GT54 untouched as independent future holdout, and avoid inventing quantitative GSI scores without suitable independent canopy labels.

## 4. Next-decision logic

- **Pipeline fails, resource or resume problems:** repair minimal loader/GPU/inference defect first; do not assume additional samples help.
- **Pipeline succeeds but appears undertrained/unstable:** only then design a limited longer, multi-image run; distinguish more optimization steps from more input images.
- **Learning runs but appears to overfit or varies markedly among the three Val scenes:** plan a **clean-only, diversely sampled, separately approved** expansion (e.g. tens of images), with frozen selection rules and rights/black-border/visual receipts. Compare within the same source generation and held Val scheme; no broad performance claim from 3 scenes.
- **Data availability becomes the measured bottleneck:** revisit the [deferred black-border recovery note](implementation/M1_5A_DATASET_INVENTORY.md) to compare valid-only crop vs validity masks vs continued HOLD, starting with a 3–4-image diagnostic. Do not reopen merely because black wedges exist.
- **GSI qualitative behavior is inadequate despite a credible source-domain pipeline:** diagnose domain shift and scale/context mismatch before choosing more OAM-TCD data, new Japan GT or an alternative model; commercial rights remain independent.

## 5. Immediate action and stopping boundary

**Next PR (G0):** promote the ten quality-reviewed Zenodo candidate images into one reproducible **small-sample-only** manifest and produce verified offline/Colab staging instructions. Review and merge this PR **before** creating the new GPU loader or running an experiment. Do not train from \`PASS_CANDIDATE\` alone. The following PR (G1) provides actual Zenodo multi-sample GPU input+one-step measurements. **No model code, threshold, new labels, data acquisition, black-border recovery or training is authorized by this design-only document.**

**Success of this plan PR:** decision order, existing source/code limitations, bounded smoke/validation and evidence hierarchy are recorded with no retroactive editing of older studies.