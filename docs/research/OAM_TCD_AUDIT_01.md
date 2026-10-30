# OAM-TCD technical audit — research notes 01

Status: **research / not an architectural decision**. Checked 2026-10-31. Sources were viewed at the upstream default branch without pinning an exact commit; pin upstream and HF revisions before implementation. No software changes or training in this research branch.

## Scope and evidence tiers
- **Paper/card:** statements made by authors.
- **Source inspected:** actual behavior of repository source at time of review, not a runtime test.
- **Open:** requires pinned revision and/or real-record verification.

## Dataset and labels
- [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/58efdd77196fa8159062afa0408245da-Abstract-Datasets_and_Benchmarks_Track.html): 5,072 RGB image tiles, nominal 2,048 px square, 0.10 m/px; >280k individual trees and >56k groups.
- [Dataset card](https://huggingface.co/datasets/restor/tcd): CC BY 4.0 subset 4,608 images; 272 NC and 192 SA are partitioned into separate repos. Full-source reported train=4,406 and test=666, whereas CC BY subset has its own split counts. **Do not conflate these figures.**
- [Zenodo](https://zenodo.org/records/11617167): `validation_fold` 0–4; -1 indicates official holdout. Official `oam_id` identifies source imagery.
- [Official generator](https://github.com/Restor-Foundation/tcd/blob/main/tools/generate_dataset.py): category IDs canopy=1, tree=2 in output COCO; optional **binary** masks are created via any nonzero annotation pixel across RGB channels. For semantic canopy/no-canopy this is a direct implementation precedent. Raw instance IDs must not be mistaken for categorical pixel values.
- **Open:** inspect a real annotation and its `segments` and `coco_annotations`; verify encoded background/void values, overlaps, image alignment, grouping and dataset revision. Source paper describes human labeling with possible omissions and inconsistent individual/tree-group boundaries.

## Training internals (source inspected)
- [COCOSegmentationDataset](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/cocodataset.py): when a transformed mask is uniformly zero or uniformly one, flips pixel (0,0). **Not acceptable for our ground-truth integrity policy without correction.**
- [COCODataModule](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/datamodule.py): random crop/augmentation; falls back to test as validation if val JSON is absent; validation transformation defaults to random crop. Must explicitly separate held-out evaluation and deterministic validation.
- [train_semantic.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/train_semantic.py): Lightning training, fixed seed 42, automatic CPU/GPU, validation tree-F1 checkpoint, grad accumulation, CSV/TensorBoard logging; `test_path` assigned to configured validation JSON in training function.
- [segformermodule.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/segformermodule.py): `SegformerConfig.from_pretrained(model_name)`, optionally pretrained weights; HF image processor with resize disabled, output logits interpolated to mask size. **Open:** confirm exact initial weights loaded for the proposed B→C comparison. Config `segformer-mit-b0.yaml` refers to backbone `nvidia/mit-b0` while specifying separately `weights: restor/tcd-segformer-mit-b0`; do not assume this initializes from Restor checkpoint.
- [Dataset generator](https://github.com/Restor-Foundation/tcd/blob/main/tools/generate_dataset.py) defaults to full dataset download and optional symlinks; unsuitable as-is for a six-tile Windows smoke test.

## Models, deployment and license
- [SegFormer MiT-B0 model card](https://huggingface.co/restor/tcd-segformer-mit-b0): 10cm binary canopy model; training involved NC images and NVIDIA research-license considerations; model card advises resampled data fine-tuning at other GSDs.
- [U-Net R34 model card](https://huggingface.co/restor/tcd-unet-r34): also indicates **CC BY-NC** weights, with future CC BY release only stated as an intention. The fact that the architecture/code may be more permissive **does not clear currently published pretrained weights**.
- [Official README](https://github.com/Restor-Foundation/tcd/blob/main/README.md): semantic SegFormer/U-Net, instance Mask R-CNN; model inference and orthomosaic tiling pipeline.
- [pyproject.toml](https://github.com/Restor-Foundation/tcd/blob/main/pyproject.toml): Apache-2.0 code, Python >=3.10 and substantial Detectron2/geospatial/Lightning deps; compare full framework versus minimal model-specific stack before choosing.

## Immediate questions (no implementation yet)
1. Pin actual repo commit, dataset revision, model revision and model-card licenses.
2. Establish exact HF schema, split sizes, folds and provenance from real CC BY records.
3. Audit end-to-end predict tiling, padding, georeferencing, output probabilities, seam behavior, evaluation metrics.
4. Compare architectures/initialization and assess clean commercial-weight pathways.
5. Reassess whether synthetic 50cm fine-tuning is the best first experiment, with explicit transfer-to-Japan limitations.

**No adoption decisions or real-data validation are claimed in this document.**
