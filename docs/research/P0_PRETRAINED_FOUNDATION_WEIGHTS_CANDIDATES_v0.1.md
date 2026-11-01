> **2026-11-01 IMPORTANT PROVENANCE UPDATE:** The published Satlas aerial training config has `Pretrained:true`, and Ai2 implementation loads torchvision `Swin_V2_B_Weights.IMAGENET1K_V1` before Satlas training. Therefore Satlas weights are **not established as ImageNet-free**. Official Ai2 ODC-BY weight release remains positive licensing evidence, but the upstream checkpoint chain-of-title needs clarification. See [P0_SATLAS_AERIAL_INITIALIZATION_LINEAGE_v0.1.md](P0_SATLAS_AERIAL_INITIALIZATION_LINEAGE_v0.1.md). This takes precedence over any unconditional commercial-clean inference below.

# P0 — Commercially usable pretrained initial weights: first-party evidence review (v0.1)

Date: 2026-11-01. **Research only; no approval to download or use any checkpoint for production.** Follows merged PRs #4 (architecture), #5 (OAM-TCD paper), #6 (scratch vs ImageNet checkpoint rights).

## 0. Core question and answer

The concern that random initialization could reduce final tree-canopy segmentation accuracy is **scientifically well grounded**. OAM-TCD fine-tuned ImageNet/MS-COCO pretrained models; its published IoUs (UNet R34 0.838, SegFormer B0 0.865) **do not quantify** how scratch initialization would fare. Pretrained features may improve convergence and final validation quality, especially for limited effective independent scenes. **But pretrained is not automatically better** after sufficiently optimized scratch training; run controlled experiments.

There is now a particularly relevant **non-ImageNet candidate**: Ai2/AllenAI **SatlasPretrain Aerial_SwinB_SI**, pretrained on **0.5–2 m/pixel aerial RGB**, a close input-domain match to GSI ~0.6m RGB. Ai2 explicitly says its foundation checkpoints are released under **ODC-BY**, with source libraries Apache-2.0. **This is much stronger positive published weight-license evidence than the unresolved torchvision ImageNet checkpoint.** However it is a Swin-v2 Base backbone, not the OAM-TCD UNet R34 or SegFormer, and 12GB GPU feasibility, transitive source/data rights and exact downloaded file provenance have not been validated.

## 1. What pretrained weights do — technical primer

- Network **architecture/code** (ResNet, U-Net, Swin) defines the computation; **weights** are tensors changed by optimization; a **checkpoint** is a saved set of those tensors and possibly optimizer state.
- **Scratch** starts with a random initialization distribution (e.g., Kaiming or Xavier), not arbitrary unconstrained noise. A meaningful network is still possible, given adequate labels/compute/training.
- **Pretraining** learns reusable representations on another task: edges/textures/objects, sometimes spatial semantics. **Fine-tuning** changes some/all of those weights using OAM canopy GT. For segmentation, pretrained encoder/backbone can feed a **newly initialized decoder/head**.
- **Frozen backbone** vs **full fine-tuning** are distinct. Freezing saves trainable compute but can under-adapt to new resolution / domain; full fine-tuning may improve accuracy but consume more VRAM.
- **Initial weights affect optimization and regularization, not just hours**; 4,608 images vs 5,072 total is not enough to infer accuracy parity when scratch vs pretrained changes.
- Match architecture/splits/resampling/augmentations/seed/optimizer/budget when testing initial weights. If comparing Satlas Swin to MONAI BasicUNet, label it a *system-level architecture-and-weights comparison*, **not an initialization-only effect**.

## 2. Evidence map, source-specific rights

| Candidate | Exact identity | Official license evidence | Domain / implementation fit | Decision |
|---|---|---|---|---|
| A. MONAI BasicUNet scratch | MONAI 1.6.1 `BasicUNet(spatial_dims=2,in_channels=3,out_channels=1)` | MONAI Apache-2.0 source; no external pretrained checkpoint | Simple, architecture differs from published OAM-TCD R34 | **Retain scratch baseline**; runtime/deps G0 unfinished |
| B. torchvision ResNet34 scratch | `resnet34(weights=None)` + independently built decoder | torchvision BSD-3-Clause code; no pretrained weight | OAM-TCD encoder family closer; decoder work required | **Comparison candidate**; runtime/deps G0 unfinished |
| C. torchvision ResNet34 ImageNet | `ResNet34_Weights.IMAGENET1K_V1`, `resnet34-b627a593.pth` | torchvision explicitly warns that weights can have terms derived from training datasets; official ImageNet dataset access is noncommercial research/education | RGB, manageable ResNet size, but origin rights ambiguous | **NOT CLEARED**; do not use until exact artifact rights are resolved |
| **D. SatlasPretrain aerial RGB** | **`Aerial_SwinB_SI`** single-image, Swin-v2-Base; also `Aerial_SwinB_MI` multi-image | Ai2 public docs explicitly say **foundation checkpoints ODC-BY**, library **Apache-2.0** | **0.5–2m/px aerial RGB** close to GSI; sizeable Swin-B and different architecture; single-image preferred for GSI | **HIGH-PRIORITY INVESTIGATION**; commercial-positive claim but artifact lineage, dependent rights and GPU not finalized |
| E. IBM/NASA Prithvi EO 2.0 | e.g. `Prithvi-EO-2.0-tiny-TL` (5M) or `100M-TL` (100M) | Official Hugging Face repos display **Apache-2.0** | Primarily **30m/px six-band HLS** images (blue/green/red/NIR/SWIR); adaptation to **0.6m three-band RGB** nontrivial | **SECONDARY**, lower immediate domain suitability |

## 3. Ai2 SatlasPretrain — first-party trace

### Precise sources

1. [Official project documentation](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) — fetched SHA `094ff770c2e2d8c4b6e2284bc9caf157d50a2087`. States **“The models are released under ODC-BY.”** Lists 0.5–2m/px aerial modality, mixed training imagery sources (US NAIP public domain, OSM ODbL, ESA WorldCover CC BY, Microsoft Buildings ODbL, etc.).
2. [Official checkpoint-loading repository](https://github.com/allenai/satlaspretrain_models/blob/main/README.md) — fetched SHA `864e7482557aac925a020614c313270c92286a77`. Explicitly lists `Aerial_SwinB_SI` and `Aerial_SwinB_MI` model IDs and their hosted files; “Checkpoints are released under ODC-BY.” Explains backbone and optional FPN; segmentation head may be **newly randomly initialized**; single-image aerial model expects **8-bit aerial RGB** and `/255` normalization.
3. [ODC-BY text in original repository](https://github.com/allenai/satlas/blob/main/DataLicense) — fetched SHA `a12f30e302933f0f2db88e7ce93186a34b9535fd`; section 3.1 explicitly permits **commercial** database use with attribution. **Important:** preamble and §2.4 say ODC-BY's grant over database structure does not automatically cover independently copyrighted underlying dataset content. Ai2 explicitly applies ODC-BY to its *model checkpoints*, a weight-level public grant; clarify its scope and any source-data restrictions before deployment.
4. [Model loader code repo LICENSE](https://github.com/allenai/satlaspretrain_models/blob/main/LICENSE) — Apache-2.0 verified; only the loader code, **not** a substitute for model checkpoint rights.
5. [Hugging Face Satlas release](https://huggingface.co/allenai/satlas-pretrain) — model/dataset card top-level `apache-2.0` badge. **Do not use this badge as the sole authority for actual weights**: first-party GitHub explicitly gives **ODC-BY for checkpoints**, **original licenses for various dataset components**, **Apache-2.0 for code**.

### Why this is attractive

- Input scale is directly relevant to GSI at ~0.6m/px — unlike 10m Sentinel-2 or 30m HLS foundational weights.
- Official checkpoint supports extracting multi-scale backbone or backbone+FPN features; possible to attach segmentation head for unioned tree canopy.
- Checkpoint-specific ODC-BY is a positive grant of commercial use (attribution conditions), **subject to provenance and third-party intellectual-property review**.
- Downside: Swin-v2 **Base** is larger than MONAI BasicUNet or ResNet34; **A2000 12GB memory/runtime unknown**. Gradient checkpointing/smaller tiles/AMP may be needed but must be tested.
- Single-image ID `Aerial_SwinB_SI` is preferred over `Aerial_SwinB_MI` for single GSI orthophotos. Do not treat multi-image checkpoint as equivalent without a test.
- No evidence yet that a Satlas aerial checkpoint is *more accurate* for Japanese urban canopy than scratch U-Net. 0.5–2m imagery match is a hypothesis-generating advantage.

### Specific unresolved questions for Satlas

- Exact HF repository revision / checkpoint SHA256, metadata, whether any other terms are attached to the file or weights.
- Detailed **lineage of the *aerial* weights**: original dataset rights, NAIP and potential other contributors, ODbL-derived annotation obligations, how Ai2 interprets their ODC-BY release for weights; consider written clarification from Ai2/appropriate internal legal review.
- Code dependencies when importing `satlaspretrain_models`; potential optional pretrained or third-party code paths; licenses and compatibility pinned.
- Actual 2D single-image GSI 3-channel tensor behavior, feature shapes, segmentation head output for 256–512px and loss; memory on RTX A2000 12GB.
- Whether trained weights can be redistributed or hosted as downstream service with required attribution and other obligations. Never equate a permissive label with a blanket warranty.

## 4. ImageNet ResNet34 — more limited grant evidence

- [Official Torchvision models and weight notice](https://docs.pytorch.org/vision/master/models.html): pretrained weights may carry separate terms from datasets and the user must ascertain permission.
- [Official ImageNet download conditions](https://www.image-net.org/download): researchers' dataset access permitted only for non-commercial research and education; original underlying images have separate owners.
- Neither source alone proves every ImageNet-derived checkpoint illegal for commercial inference; equally, torchvision **BSD code** does not clear the **checkpoint**.
- Therefore do not promote `IMAGENET1K_V1` to commercial path without specific written/explicit assurance for the exact source model.

## 5. Prithvi EO 2.0 — why not first

- [Official IBM/NASA model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL), Apache-2.0 badge and 30m 6-band HLS pretraining.
- A **30m multispectral** encoder changes both input spectral bands and feature spatial scales relative to GSI 0.6m RGB, requiring first-layer adaptation, positional embedding/shape accommodation and new segmentation head. It might generalize but this is unproven and a less direct first option.
- `Prithvi-EO-2.0-tiny-TL` (5M) exists as a compute-conscious backup if Satlas Base proves too heavy; **not yet vetted as a commercial-clean weight at individual file level**.

## 6. Fair experiments (after separate approval and G0/G1 gates)

**E0 baseline:** MONAI BasicUNet scratch (or chosen ResNet34 scratch) on **CC BY-only OAM-TCD**, `oam_id` group split, fixed seed/validation and target ~0.5–0.6m degradation. Record loss/IoU/F1/precision/recall/area-rate bias/compute.

**E1 within-architecture comparison (best causal test if checkpoint rights cleared):** independent ResNet34 U-Net with `weights=None` vs `weights=...pretrained`, identical architecture/decoder and same training recipe. ImageNet V1 rights currently **not cleared**; do not run as a commercial-cleared result now.

**E2 system-level comparison:** Satlas `Aerial_SwinB_SI` pretrained backbone with newly initialized segmentation head vs scratch U-Net. Different models: evaluate *best usable system performance*, not initial-weight effect. If feasible, run **Satlas Swin-B scratch vs Satlas Swin-B pretrained** with same architecture as a controlled comparison, budget permitting.

Compare at equal training **steps**, **GPU-hours**, and/or until common convergence criteria, declaring which comparison; record independent Japan GSI qualitative examples but reserve GT54 for later final independent stage. No unsupported accuracy predictions.

## 7. Recommendation and gate

- **Upgrade SatlasPretrain aerial Swin-v2 Base to first-priority pretrained investigation**, while **keeping random-initialization MONAI BasicUNet as the safe engineering reference**.
- **Do not force a premature choice** between random and pretrained. User's concern about final accuracy is valid; use empirical trials and provenance audit.
- **No G0 pass:** package wheel / transitive dependency notices, checkpoint SHA/revision, license and legal interpretation, true pixel QA, GPU feasibility, and training/evaluation metrics remain **NOT VERIFIED**.
- Before implementation, decide whether to pursue an Ai2 clarification of weight-commercial permissions due mixed-origin annotation terms. Do not represent this note as legal advice.
- As of this PR, no download, model execution, changed baseline, or approved pretrained checkpoint use occurred.

## Source citation ledger (direct first-party)

- Ai2 foundation-model license and dataset provenance: https://github.com/allenai/satlas/blob/main/SatlasPretrain.md
- Ai2 loader/checkpoint IDs and segmentation heads: https://github.com/allenai/satlaspretrain_models/blob/main/README.md
- Ai2 weight license text: https://github.com/allenai/satlas/blob/main/DataLicense
- Ai2 Hugging Face catalog: https://huggingface.co/allenai/satlas-pretrain
- Torchvision warning: https://docs.pytorch.org/vision/master/models.html
- ImageNet access terms: https://www.image-net.org/download
- IBM/NASA Prithvi example: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL
