# OUEM Canopy Segmentation

Research repository for RGB-based **semantic canopy segmentation** intended to support simplified 3D canopy models in the Open Urban Environment Model (OUEM).

The near-term experiment asks whether fine-tuning a pretrained canopy model on synthetically degraded **50 cm/pixel** imagery improves segmentation compared with the unchanged model. Individual-tree delineation is not required.

## Project documents
- [Design principles](docs/DESIGN_PRINCIPLES.md)
- [Experiment plan](docs/EXPERIMENT_PLAN.md)
- [OUEM handoff requirements](docs/OUEM_HANDOFF.md)

## Status
**Foundation / Phase 1 planning.** No validated real-data training or accuracy results are claimed. The P1-1 implementation is being developed separately in a draft PR; it must be checked against these documents before merge.

## License caution
The pretrained `restor/tcd-segformer-mit-b0` model weights have noncommercial restrictions. Fine-tuning does not remove them. Research artifacts are not a commercially deployable OUEM model.

## History
This repository was formerly `oam-tcd-poc`. Earlier inference and scale/edge-effect experiments remain accessible in Git history, even when removed from the active tree.
