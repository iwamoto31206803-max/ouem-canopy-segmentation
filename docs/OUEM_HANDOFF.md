# OUEM Model Handoff — planning baseline

Status: requirements, not a frozen API or validated model package.

## Responsibilities
This repository owns RGB segmentation model research, training, metrics, provenance, and reproducible inference settings. OUEM owns geospatial image acquisition, CRS and GSD handling, wide-area tiling, mask georeferencing, polygonization, CHM integration, simplified 3D canopy and environmental evaluation.

## Model package requirements
- Architecture, model and checkpoint revision, checksum, dependency versions, seed, experiment ID and training config.
- Dataset and annotation lineage, split grouping, source hashes, evaluation and license restrictions.
- Input: RGB order, dtype, range, dimensions, GSD assumptions, normalization, resizing and nodata policy.
- Output: logits/probability/mask semantics, class mapping, threshold, tiling, overlap, padding and postprocessing.
- Golden input/output inference fixture with checksums and documented tolerances.
- Metrics: IoU, F1, precision, recall, canopy area bias, patch and boundary performance, domain-shift failures.

## Integration acceptance
OUEM must replicate preprocessing and model-specific inference semantics. Tile seams, spatial transforms and polygons are validated downstream, not inferred from model-only tests. Noncommercial pretrained weights and fine-tuned derivatives must not be labelled commercially deployable. Nationwide generalization remains unproven until independent evaluation.
