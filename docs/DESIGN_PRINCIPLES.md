# Design Principles — OUEM Canopy Segmentation

**Status:** Proposed v0.2 baseline, 2026-11-01, subject to design PR review/merge.  
**Authority:** [Project Charter](PROJECT_CHARTER.md) defines purpose, scope, success and phases; [ADR: Satlas-first](decisions/ADR_20261101_SATLAS_FIRST.md) records the new engineering direction. Historical `EXPERIMENT_PLAN.md` and earlier research records cannot override this approved direction once merged.

## Mission and boundaries

- **Build a usable component, not conduct comparative research.** Develop a reproducible RGB semantic canopy model suitable for OUEM's simplified 3D canopy pipeline using Japanese GSI aerial imagery at roughly **0.5–0.6 m/pixel**. Aim for **practically adequate** accuracy and license clarity rather than maximal published metrics.
- This repository owns source-data and label preparation, trained canopy model, fixed inference normalization/semantics, provenance, QA and technical evaluation.
- OUEM owns imagery acquisition, CRS/georeferencing, large-area tiling, vectorization, **joining the canopy mask with point-cloud data to produce height-distribution/3D canopy data**, and subsequent shade/thermal/landscape evaluation and its effectiveness assessment. This repository is accepted on **2D canopy-mask quality, stable inference and spatial handoff**, not on downstream point-cloud height accuracy or environmental simulation outcomes.
- Single-tree identification is **not a requirement**. CHM-assisted segmentation is **not** a prerequisite.
- The user does **not** plan to publish or redistribute learned weights or a resulting model. Future **internal/commercial inference** is the use case; changes in deployment scope require separate rights review.


- **Defer the numeric target GSD.** Before M1 data preprocessing, inspect existing GSI GeoTIFF CRS, ground pixel size, source tile zoom, and reprojection/resampling. Do not equate EPSG:3857 nominal pixel spacing with Japanese ground resolution or original-photo native resolution. Select a pragmatic training GSD after this check; do not pre-plan a 0.5m/0.6m comparison.

## Preferred engineering route

**Prioritize:** Ai2 `Aerial_SwinB_SI` SatlasPretrain checkpoint -> official compatible pretrained backbone and pretrained FPN as the initial choice -> newly initialized binary segmentation head -> Fine-tuning with **verified CC BY 4.0 OAM-TCD** labels and approximately GSI-scaled inputs -> verify on real GSI images -> handoff to OUEM.

- This **does not** require original Restor pretrained weights, nor a Swin-v2 Base scratch control.
- Do not claim that matched spatial resolution guarantees Japan/GSI performance.
- `Head.SEGMENT` is the initial two-class head choice; exact label/ignore semantics, package compatibility, crop/GSD choices, freeze policy, threshold, license notices and compute budget are **open design details**, to be settled by the smallest informative checks.
- If blocked by memory, licensing, dataset quality or GSI model quality, diagnose and propose a bounded fallback rather than launch an open-ended architecture contest.

**Agreed initial model composition:** official `Aerial_SwinB_SI` pretrained Swin-v2 Base backbone + pretrained FPN + official `Head.SEGMENT` two-class semantic head newly initialized for two classes (background, Tree/Canopy), starting with official cross-entropy loss. Confirm loader API, normalization, class-index target shapes, cross-entropy loss and void/ignore handling, and successful forward/backward smoke. The official path uses FPN/upsampling and a simple convolutional head, not an interchangeable generic U-Net decoder; measure GPU memory before choosing fine-tuning scope. Do not develop a custom decoder unless a specific blocker warrants it.

## Non-negotiable principles

1. **Milestone first; avoid premature validation burden.** **M1** is completion of SatlasPretrain-initialized fine-tuning with verified OAM-TCD CC BY-only data, basic OAM-TCD validation, a saved checkpoint and successful canopy-mask inference. **M2** is qualitative inspection on several existing GSI scenes, followed by problem-driven work only if needed. Do **not** require new GSI GT, predefined Japanese accuracy thresholds, exhaustive comparisons or nationwide validation for M1/M2. Visual plausibility must not be reported as quantified accuracy.
2. **Explicit evidence levels.** Design choice ≠ successful model import ≠ synthetic smoke ≠ real-data train ≠ source-domain validation ≠ independent Japan/GSI evaluation ≠ operational suitability. Track each separately; do not invent PASS.
3. **No data leakage.** Split by `oam_id`/source scene group. Keep official OAM-TCD holdout out of development tuning. Keep GT54 independent and **verify label suitability** before using its land-cover map as canopy GT.
4. **Input-label integrity.** Verify source license at the record level and panoptic IDs/categories. Explicitly map Tree + Canopy positive, with background/void handling and uniform masks preserved unchanged; never edit (0,0) for a loader shortcut.
5. **Reproducible resolution adaptation.** Record nominal source GSD, exact crop, downscale/antialiasing, RGB rounding, GT fractional coverage and binarization threshold; make RGB/GT grids coincide. The former 10cm->50cm factor-5 recipe is a **candidate**, not proof of 50cm native imaging equivalence or a compulsory A/B/C experiment.
6. **License evidence and bounded commercial risk.** Separate (a) code/dependencies, (b) starting checkpoint, (c) data/annotations, (d) resulting checkpoint/use. Official Satlas checkpoint ODC-BY is positive evidence, **not complete legal assurance**; record attribution and unresolved ImageNet upstream rights. Restrict training inputs to verified CC BY 4.0 OAM-TCD; **do not** use restricted Restor pretrained checkpoints in commercial-candidate training. Do not conflate a package's Apache/BSD license with its weights.
7. **One pragmatic route, controlled expansion.** Satlas-first implementation is preferred; no exhaustive model rankings, causal pretraining ablation or large-scale labels by default. Add work only when a specific acceptance blocker demonstrates necessity.
8. **Compute and environment transparency.** Verify official load paths/heads and local CPU tests before GPU work. Benchmark patch size/VRAM and appropriate fine-tuning freeze strategy rather than assuming an RTX A2000-class GPU can handle full-res training.
9. **Stable model handoff.** Lock input RGB range/order/GSD/normalization, valid-area behavior, output probabilities, threshold and tiling/padding/overlap. Version weights with checksum and config; OUEM verifies geospatial seams, CHM coupling and polygon output separately.
10. **Small and reversible.** Keep implementation isolated, preserve old research and Git history, prefer bounded/offline-friendly tests and small reviewed PRs. Do not merge unfinished technical work into `main`.
11. **Proportionate controls, with a stop rule.** A new gate, audit or isolation mechanism needs a named risk, the least expensive credible check, a completion criterion and a reason simpler existing safeguards are insufficient. Do not make obtaining absolute certainty a prerequisite for a bounded research experiment; do not mistake a completed audit for model progress. Reassess recurring gates when their cost delays M1/M2 outcomes.

## Risk-proportionate execution policy (2026-11-14 proposal)

**Default:** Use the established host Python/VS Code/Codex/GitHub workflow for ordinary code development, metadata analysis, synthetic tests and non-destructive research processing. **Windows Sandbox is optional, not a project-wide prerequisite.** An OS-enforced zero-network requirement is likewise **not a blanket condition** imposed by CC BY 4.0 or by this repository's research mission. Choose network isolation when a concrete threat, restricted data, organizational policy or untrusted executable actually warrants it; then test the selected control.

**Two evidence levels, not two licenses:**

- **Bounded exploratory research:** A small, expressly scoped real-data experiment may be approved with proportionate evidence: permitted source/revision and attribution, Train/Val/test separation, defensible image/label/coverage limitations, read-only inputs, fresh outputs and a reproducible run record. Uncertain scenes may stay excluded rather than trigger indefinite forensic investigation. Record uncertainty and disallow unverified commercial or public claims. **This does not automatically admit data to R1.**
- **Formal R1 dataset / production-facing artifacts:** Preserve the adopted R1a verifier, genuine source-bound rights and review assertions, data integrity and QA, and independent acceptance before declaring R1 PASS. An exploratory run does not waive these requirements or make a HOLD sample PASS.

**Transition safety:** Historical R1d-1B-5/6 decision records and Issue #64 retain their provenance. The four-image real execution route is currently `REAL_EXECUTION_DISABLED`; G3/G4 human decisions remain PENDING and all 150 formal R1 candidates remain HOLD. **This documentation-only policy does not grant real-data execution, change existing code gates, retroactively approve `ALL_VALID` or authorize processing under a new mode.** Before using a simpler host-based real-data path, review and expressly approve a bounded replacement execution contract and its minimal code/test changes. Do not silently bypass an existing enforced prohibition.

**Review budget:** Reuse already verified receipts and tests; do not restart completed QA or create a new framework for a single pilot. Require additional diagnostics only when they can change a concrete data-selection, model-training, safety or rights decision. When the direct next experiment is blocked by unavailable hardware, continue independent software/data-planning tasks rather than expanding the blocker into project-wide restrictions.

## Change / acceptance discipline

- First accept the **mission and Satlas-first architecture direction** in documentation. Then commission minimal implementation compatibility review and progress to **M1 fine-tuning completion**, followed by **M2 lightweight GSI visual review**. Do not treat this documentation PR as license G0/P1 data QA or numerical accuracy PASS. More demanding GSI GT/quantitative evaluation is **deferred until a demonstrated need**, not assumed as an initial milestone.
- Prior historical G0/G1/G2/G3 research gates remain **records**, not requirements to run three competing models. See the current D0–D5 engineering stages in the [Project Charter](PROJECT_CHARTER.md).
- Requirements changes should state source/evidence, what they supersede, intended operational consequences, responsible repository and reversibility.
- Record validated results, limitations and failure modes; no accuracy or commercial-deployment statement without appropriate evidence.
