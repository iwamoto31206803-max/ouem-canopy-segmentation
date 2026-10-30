# OAM-TCD technical audit — research notes 03: model families and licenses

Status: **evidence review, not legal advice, runtime test, or architecture decision**. Prepared 2026-10-31. Source versions must be pinned before acceptance. This document deliberately distinguishes software, training data, pretrained weights and trained derivatives.

## First-party sources consulted

1. [Restor TCD README](https://github.com/Restor-Foundation/tcd/blob/main/README.md): project code Apache-2.0; Mask R-CNN / U-Net / SegFormer available; all currently released models described as CC BY-NC 4.0, with CC BY re-training planned. NVIDIA SegFormer research-license warning; SMP U-Net library MIT.
2. [Restor dataset card](https://huggingface.co/datasets/restor/tcd): `restor/tcd` is CC BY 4.0 (4,608 images); separate `restor/tcd-nc` (272) and `restor/tcd-sa` (192). Published models trained on CC BY plus NC subset, making data licensing different from checkpoint licensing.
3. [Restor U-Net R34 model card](https://huggingface.co/restor/tcd-unet-r34/blob/main/README.md): explicitly says `License: CC-BY-NC; CC-BY to follow`. No evidence reviewed that a cleared CC BY release currently exists.
4. [Restor SegFormer MiT-B0 model card](https://huggingface.co/restor/tcd-segformer-mit-b0): states SegFormer research-license restrictions and trained on NC imagery; the Hugging Face interface license badge is broad/ambiguous (`cc`). Do **not** infer a permissive checkpoint license from the badge.
5. [NVIDIA original SegFormer repository](https://github.com/NVlabs/SegFormer): README states research/evaluation noncommercial terms; [license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) restricts its Work and derivatives, but **scope must be traced** before asserting the same terms bind independently implemented architectures.
6. [NeurIPS 2024 OAM-TCD supplemental](https://proceedings.neurips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Supplemental-Datasets_and_Benchmarks_Track.pdf), section 2.1 (publication states): Mask-RCNN and U-Net weights CC BY-NC 4.0; SegFormer weights under NVIDIA Source Code License for SegFormer. Treat this as more specific than generic README language; clarify any remaining inconsistency with provider before commercial deployment.

## Three independent rights questions
| Layer | Baseline evidence | Consequence |
| --- | --- | --- |
| TCD pipeline software | Apache-2.0, but dependent packages have separate terms | Inspect installed dependencies and copied source. |
| Training and evaluation images/GT | Only the `restor/tcd` CC BY portion for a permissive-data route | Maintain per-record attribution, provenance and no NC/SA mixing. |
| Initial and final model weights | Restor published U-Net/Mask-RCNN NC and SegFormer NVIDIA-specific restriction | Fine-tuning does not automatically remove upstream restrictions. Audit all initialization checkpoints, including ImageNet backbone weights. |

Note: an architecture described in a paper is not itself necessarily encumbered by the identical terms as a particular code implementation or published checkpoint. The NVIDIA license scope and other rights must be assessed before saying a clean SegFormer implementation is or is not commercially possible. Do not copy NVIDIA code or use NVIDIA-restricted weights in a commercial-cleared route without appropriate permission.

## Model-family fit (preliminary, not ranked accuracy)
- **SegFormer MiT-B0:** existing user PoC and Restor baseline, practical reference for 50cm transfer experiments, Hugging Face implementation available; weight lineage and exact initialization must be recorded.
- **U-Net ResNet34:** supports simple pixelwise binary canopy mapping; feasible independent CC BY-only training hypothesis if source code, encoder initialization and trained artifacts have cleared rights. Published Restor U-Net weights themselves remain NC.
- **Mask R-CNN:** instance pipeline adds complexity not needed for the current continuous canopy semantic output; do not exclude it for future object tasks.
- **Alternative backbones:** could be evaluated for real deployment after baseline validity, but benchmarking before clean labels/splits would be premature.

## Candidate experiment architecture — not adopted
**Track R (research benchmark):** existing Restor pretrained model, with restrictions explicitly carried; reproduce current observations and evaluate controlled resolution adaptation. No licensing claim beyond permissions.

**Track C (cleared from inputs):** only source imagery + labels with appropriate rights, independently cleared framework and initialization checkpoints, reproducible training and provenance; commercial application remains subject to independent legal and operational checks. No automatic assumption that training with CC BY data alone yields a commercially deployable result.

Track R and C must not share learned parameters if the commercial-clean claim depends on independence. Hyperparameter ideas and published findings can be compared separately while tracking provenance.

## Gaps before a decision
1. Pin exact HF model repos/commits, inspect model-card YAML license vs prose and original training source history; determine whether any official CC BY model checkpoints were actually released after README statements.
2. Inspect license of each implementation and encoder pretraining source for clean-run candidates; test assumptions against legal/compliance requirements.
3. Contrast actual SegFormer B0 vs U-Net R34 parameter count, speed/VRAM and segmentation quality at 10cm and 50–60cm on identical independent records; no unmeasured performance claims.
4. Confirm official TCD CC BY subset split counts and actual panoptic schema; perform record-level tests and assess domain shift to GSI.
5. Revisit initial A/B/C 50cm plan only after provenance and evaluation design are settled.

### Decision status
No model selected. No commercial-cleared pretrained weights established. Preserve current `EXPERIMENT_PLAN.md` as a hypothesis, not fixed implementation requirements.
