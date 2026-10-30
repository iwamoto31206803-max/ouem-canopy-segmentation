# Design Principles — OUEM Canopy Segmentation

Status: proposed baseline (2026-10-30). Scope: canopy semantic segmentation model R&D; OUEM integration is downstream.

## Mission and boundaries
- Build a reproducible RGB-only canopy segmentation model that can eventually support nationwide simplified 3D canopy modelling from public aerial imagery.
- This repository owns data preparation, label QA, training, evaluation, model artifacts and their specifications.
- OUEM owns geographic imagery acquisition, georeferencing, wide-area inference, polygons, CHM/height integration, 3D canopy, and environmental analyses.
- Individual tree identification and CHM-assisted segmentation are not Phase 1 requirements.

## Non-negotiable principles
1. **Validity over attractive maps.** No claim of accuracy without independent ground truth; report false positives, omissions, area bias, boundary errors and failure cases.
2. **No leakage.** Split at original OAM image/group (`oam_id`) level. Keep the official holdout and Japanese GT54 untouched for Phase 1 development. GT54 land-cover labels are not automatically canopy ground truth.
3. **Reproducibility.** Record source dataset revision and licenses, image/label hashes, group/fold split, seeds, pixel geometry, preprocessing, environment, metrics and model configuration. Preserve raw inputs unchanged.
4. **Label integrity.** Decode panoptic segment IDs using verified segment-to-category metadata; Tree and Canopy are positive classes only when explicitly mapped. Never modify all-zero/all-one masks to satisfy training code.
5. **Explicit resolution.** Nominal Phase 1 10 cm→50 cm is a controlled degradation, not evidence of true 50 cm sensor equivalence. Keep crop, resampling, GSD assumptions and image/GT alignment auditable.
6. **Licensing is a gate.** Upstream code, dataset, model weights and architecture may have different terms. The existing `restor/tcd-segformer-mit-b0` weights are noncommercial; fine-tuning does not remove that restriction. Commercial deployment requires a separately cleared solution.
7. **Separate evidence levels.** Distinguish design, implementation, synthetic tests, real-data verification, trained-model evaluation and nationwide generalization. No invented PASS.
8. **Small and reversible.** Prefer minimal dependencies, offline CPU tests, small PRs and simple modules. Keep experiments traceable through Git history rather than retaining obsolete scripts in the active tree.
9. **Stable model handoff.** Model-specific normalization, resizing, thresholds and output semantics must be explicit and reproducible in OUEM.

## Change discipline
- Change experiments through a documented PR with rationale and effect on comparability.
- Do not merge unfinished work into `main`. Do not delete or rewrite Git history to clean up prototypes.
- This document records principles, not a claim that every current script complies.
