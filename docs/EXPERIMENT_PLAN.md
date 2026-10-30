# Experiment Plan — canopy adaptation to 50 cm

Status: proposed baseline; no completed Phase 1 training or validated accuracy claim.

## Research question
Can fine-tuning a pretrained RGB canopy segmentation model on synthetically degraded 50 cm/pixel TCD imagery improve predictions at this resolution, relative to the unchanged pretrained model?

## Fixed comparison
- **A:** pretrained `restor/tcd-segformer-mit-b0` on original ~10 cm imagery.
- **B:** same pretrained model on the degraded ~50 cm imagery.
- **C:** fine-tuned model on the *same* degraded ~50 cm evaluation imagery used for B.
- Primary adaptation contrast: **B→C**. A is contextual, not an equal-resolution baseline.
- Use fixed evaluation records, masks and thresholds when comparing B/C; log any threshold tuning on validation only.

## P1-1: data and preprocessing
- Source: `restor/tcd` CC BY 4.0 dataset; verify actual dataset revision, row schema, class/category IDs and panoptic segment mapping before processing.
- Preserve raw imagery and labels. Convert Tree + Canopy into a binary mask at native resolution through verified metadata.
- For a nominal 2048×2048 source at ~0.10 m/pixel, top-left crop to 2040×2040, factor-5 reduce to 408×408 at ~0.50 m/pixel.
- RGB: area/block averaging with a documented rounding rule. GT: per 5×5 block canopy fraction, binary positive at fraction >=0.5; retain fractional GT separately.
- Validate source dimensions, orientation, alignment, category coverage and masks with all-zero/all-one cases.
- Initial small real-data subset: 4 train and 2 validation examples, grouped by `oam_id`, chosen deterministically from official training folds; no holdout reads.
- Manifest: dataset version/revision, license, source identifiers, source SHA-256, group/fold assignment, transforms, output hashes, dimensions, seed, selection algorithm and QA findings.
- Synthetic/offline tests do not constitute real-data acceptance.

## P1-2: train and evaluate
- Run CPU smoke test, then GPU fine-tuning; preserve exact checkpoint initialization and training config.
- Report Tree IoU, F1, Precision, Recall; also canopy area fraction bias, small-patch recall and boundary quality.
- Check interruption/restart determinism and checkpoint validity.
- Avoid upstream loader behavior that alters uniform masks; ensure validation and final holdout roles are distinct.
- Qualitative GSI tests are supplemental, not proof of generalization.

## Later phases
- **Phase 2:** Japan/GSI domain shift, imagery dates, vegetation types, shadows, mixed landscapes; ~0.597 m/pixel GSI tests and optional 60 cm sensitivity after the main 50 cm comparison.
- **Phase 3:** independent Japanese evaluation; assess suitability and label correspondence of GT54 before any use. GT54 is not part of Phase 1.

## Acceptance and decisions
P1-1 passes only after real dataset decoding, split integrity, source provenance and alignment are checked. P1-2 passes only after reproducible B/C evaluation with recorded limitations. Changes to splits, resampling, label definitions or metrics require explicit rationale and versioning.
