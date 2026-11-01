# OUEM Canopy Segmentation

Engineering repository for a **practical RGB tree-canopy semantic-segmentation component** of the **Open Urban Environment Model (OUEM)**.

**Goal:** Starting from widely available Japanese GSI RGB aerial orthophotos (approximately **0.5–0.6 m/pixel**), produce **good-enough** tree-canopy masks for OUEM's simplified 3D canopy, heat/shade and landscape workflows. The aim is **usable accuracy, maintainable implementation and a reasonably documented path to internal commercial inference**, **not** a publication or proof of state-of-the-art performance.

## Current direction: SatlasPretrain-first (design proposal)

- Preferred initialization: Ai2 SatlasPretrain aerial RGB single-image **`Aerial_SwinB_SI`** (Swin-v2 Base, with optional pretrained FPN).
- Supervision: **OAM-TCD CC BY 4.0-only** RGB/Tree+Canopy segmentation labels, after revision and per-record license/annotation QA.
- Fine-tuning target: imagery resampled to approximately **GSI's 0.5–0.6m/px** scale; exact training grid/resampling to be fixed before training.
- Output: binary semantic canopy probabilities/masks and reproducible model input/output contract; **not individual-tree delineation**.
- **No public redistribution or publication of the trained model/checkpoint is planned.** Internal training and possible commercial inference are the intended scope.
- A full scratch-vs-pretrained Swin experiment is **not required**. Practical fallback is available only if this route fails.

This is a **preferred candidate and design direction**, **not** a trained or validated production model. No claim of GSI accuracy, GPU feasibility or definitive commercial rights clearance is made.

## Source of truth

1. [Project Charter: purpose, scope, success and staged gates](docs/PROJECT_CHARTER.md)
2. [Design Principles: mandatory engineering rules](docs/DESIGN_PRINCIPLES.md)
3. [Satlas-first decision record](docs/decisions/ADR_20261101_SATLAS_FIRST.md)
4. [OUEM handoff requirements](docs/OUEM_HANDOFF.md)
5. [Prior experiment plan — historical / superseded](docs/EXPERIMENT_PLAN.md)
6. [Research evidence, provenance and license investigations](docs/research/)

**Status:** Design rebase in progress (2026-11-01). The first PR is **documentation-only**. Before training, perform official model/head loading QA, verify the exact OAM-TCD CC BY training records and transforms, and establish viable GPU and evaluation conditions. Old research decisions remain informative but are not controlling when explicitly superseded by the charter and ADR.

## License and evaluation caution

Ai2 explicitly publishes SatlasPretrain checkpoint weights under **ODC-BY**; its model-loading library is **Apache-2.0**. The exact upstream ImageNet chain and commercial production obligations remain a documented review item, **not proof of prohibited use or final signoff**. Dataset labels/images must be restricted to verified **CC BY 4.0** OAM-TCD records with attribution; pretrained Restor noncommercial/restricted released weights must **not** be copied into the intended commercial-use training chain.

Do not present model visualizations, historical 10cm OAM-TCD benchmark scores or past scale-up observations as demonstrated GSI accuracy. Quantitative claims need independent target-domain canopy reference labels; GT54 is reserved and its land-cover classes are not automatically canopy GT.

## Project history

This repository was formerly `oam-tcd-poc`. Prior model inference, scale/edge-effect experiments, scratch-route investigations and the original OAM-TCD adaptation A/B/C plan remain available in Git history and historical documents. They are **not** the current implementation mandate.
