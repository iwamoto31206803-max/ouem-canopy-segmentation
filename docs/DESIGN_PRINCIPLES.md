# Design Principles — OUEM Canopy Segmentation

**Status:** Proposed v0.2 baseline, 2026-11-01, subject to design PR review/merge.  
**Authority:** [Project Charter](PROJECT_CHARTER.md) defines purpose, scope, success and phases; [ADR: Satlas-first](decisions/ADR_20261101_SATLAS_FIRST.md) records the new engineering direction. Historical `EXPERIMENT_PLAN.md` and earlier research records cannot override this approved direction once merged.

## Mission and boundaries

- **Build a usable component, not conduct comparative research.** Develop a reproducible RGB semantic canopy model suitable for OUEM's simplified 3D canopy pipeline using Japanese GSI aerial imagery at roughly **0.5–0.6 m/pixel**. Aim for **practically adequate** accuracy and license clarity rather than maximal published metrics.
- This repository owns source-data and label preparation, trained canopy model, fixed inference normalization/semantics, provenance, QA and technical evaluation.
- OUEM owns imagery acquisition, CRS/georeferencing, large-area tiling, vectorization, CHM/height and simplified 3D geometry, shade/thermal and landscape analyses.
- Single-tree identification is **not a requirement**. CHM-assisted segmentation is **not** a prerequisite.
- The user does **not** plan to publish or redistribute learned weights or a resulting model. Future **internal/commercial inference** is the use case; changes in deployment scope require separate rights review.

## Preferred engineering route

**Prioritize:** Ai2 `Aerial_SwinB_SI` SatlasPretrain checkpoint -> official compatible pretrained backbone and potentially FPN -> newly initialized binary segmentation head -> Fine-tuning with **verified CC BY 4.0 OAM-TCD** labels and approximately GSI-scaled inputs -> verify on real GSI images -> handoff to OUEM.

- This **does not** require original Restor pretrained weights, nor a Swin-v2 Base scratch control.
- Do not claim that matched spatial resolution guarantees Japan/GSI performance.
- Exact head semantics, package compatibility, crop/GSD choices, freeze policy, threshold, license notices and compute budget are **open design details**, to be settled by the smallest informative checks.
- If blocked by memory, licensing, dataset quality or GSI model quality, diagnose and propose a bounded fallback rather than launch an open-ended architecture contest.

## Non-negotiable principles

1. **Usability over publishable comparisons.** Define a bounded success criterion from OUEM workflow needs: canopy coverage area bias, missed trees/canopy, building/shadow false positives, landscape failure cases, acceptable processing cost and reproducibility. **No arbitrary fixed IoU requirement is claimed yet.** Accuracy may not be claimed from an attractive map alone.
2. **Explicit evidence levels.** Design choice ≠ successful model import ≠ synthetic smoke ≠ real-data train ≠ source-domain validation ≠ independent Japan/GSI evaluation ≠ operational suitability. Track each separately; do not invent PASS.
3. **No data leakage.** Split by `oam_id`/source scene group. Keep official OAM-TCD holdout out of development tuning. Keep GT54 independent and **verify label suitability** before using its land-cover map as canopy GT.
4. **Input-label integrity.** Verify source license at the record level and panoptic IDs/categories. Explicitly map Tree + Canopy positive, with background/void handling and uniform masks preserved unchanged; never edit (0,0) for a loader shortcut.
5. **Reproducible resolution adaptation.** Record nominal source GSD, exact crop, downscale/antialiasing, RGB rounding, GT fractional coverage and binarization threshold; make RGB/GT grids coincide. The former 10cm->50cm factor-5 recipe is a **candidate**, not proof of 50cm native imaging equivalence or a compulsory A/B/C experiment.
6. **License evidence and bounded commercial risk.** Separate (a) code/dependencies, (b) starting checkpoint, (c) data/annotations, (d) resulting checkpoint/use. Official Satlas checkpoint ODC-BY is positive evidence, **not complete legal assurance**; record attribution and unresolved ImageNet upstream rights. Restrict training inputs to verified CC BY 4.0 OAM-TCD; **do not** use restricted Restor pretrained checkpoints in commercial-candidate training. Do not conflate a package's Apache/BSD license with its weights.
7. **One pragmatic route, controlled expansion.** Satlas-first implementation is preferred; no exhaustive model rankings, causal pretraining ablation or large-scale labels by default. Add work only when a specific acceptance blocker demonstrates necessity.
8. **Compute and environment transparency.** Verify official load paths/heads and local CPU tests before GPU work. Benchmark patch size/VRAM and appropriate fine-tuning freeze strategy rather than assuming an RTX A2000-class GPU can handle full-res training.
9. **Stable model handoff.** Lock input RGB range/order/GSD/normalization, valid-area behavior, output probabilities, threshold and tiling/padding/overlap. Version weights with checksum and config; OUEM verifies geospatial seams, CHM coupling and polygon output separately.
10. **Small and reversible.** Keep implementation isolated, preserve old research and Git history, prefer bounded/offline-friendly tests and small reviewed PRs. Do not merge unfinished technical work into `main`.

## Change / acceptance discipline

- First accept the **mission and Satlas-first architecture direction** in documentation. Then commission minimal implementation compatibility review. Do not treat this documentation PR as license G0/P1 data QA or numerical accuracy PASS.
- Prior historical G0/G1/G2/G3 research gates remain **records**, not requirements to run three competing models. See the current D0–D5 engineering stages in the [Project Charter](PROJECT_CHARTER.md).
- Requirements changes should state source/evidence, what they supersede, intended operational consequences, responsible repository and reversibility.
- Record validated results, limitations and failure modes; no accuracy or commercial-deployment statement without appropriate evidence.
