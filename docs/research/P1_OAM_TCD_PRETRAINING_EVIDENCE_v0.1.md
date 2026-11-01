# P1 Research — OAM-TCD initialization and scratch-training evidence (v0.1)

Date: 2026-11-01. **Research only**. This follows merged PR #4 and does not change implementation, trained models, licensing gate or validated metrics.

## Finding — OAM-TCD did not train the public reference models from random initialization

The **main NeurIPS 2024 OAM-TCD paper**, section 4 “Baseline Models,” explicitly states that **in all cases the authors fine-tuned models previously trained on ImageNet or MS-COCO** and that this enabled faster convergence than starting from scratch. Main source: https://openreview.net/pdf/9c48222698818db2bebb7bfa7ee994f2785bbe6b.pdf (section 4; PDF printed p. 7). The **official supplement**, “Model Card(s)”, independently states **fine-tuning / transfer learning from ImageNet and MS-COCO; otherwise OAM-TCD was the only training data**: https://papers.nips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Supplemental-Datasets_and_Benchmarks_Track.pdf (section 2.1, printed p. 9).

**Strongly established:** pretrained initialization was used for the paper's baseline models. **Not established here:** the exact encoder pretrained checkpoint, individual weight license, and per-family initialization setting (UNet R34/R50 vs SegFormer B0–B5 vs Mask-RCNN). Do not infer they all used identical weight sources.

**Critical distinction:** using ImageNet **normalization statistics** is not evidence of pretrained weight use; the paper explicitly mentions both, and they are separate choices.

## Source-provided experimental procedure

| Element | Authors' reported baseline | Consequence for OUEM PoC |
|---|---|---|
| Training data | OAM-TCD reference release, including some CC BY-NC images; CC BY-SA test-only | OUEM commercial-clean training uses CC BY 4.0 subset only |
| Images | 5072 total OAM-TCD images; CC BY repository contains 4608 | ~9.1% lower total dataset count, **not** necessarily the exact training-split reduction |
| Initialization | ImageNet/MS-COCO pretrained | OUEM working baseline: random initialization |
| Models | UNet ResNet34/50, SegFormer MiT-B0–B5, Mask-RCNN R50 | OUEM candidate MONAI BasicUNet or torchvision R34 + independently developed U-Net decoder |
| Input / cropping | Source 2048² @ nominal 10 cm/px, random 1024² crops | Need distinguish patching from 10→50–60 cm/px GSD degradation |
| Color/geometry | ImageNet normalization; random rotation, horizontal/vertical flips, color transforms | Match reproducible preprocessing where useful |
| Optimization (semantic) | AdamW, 1e-4 to 1e-5, Reduce-on-plateau, 32 batch, 75 CV epochs / 50 final epochs | Baseline guidance only, **not** a validated scratch-learning schedule |
| Loss | focal (UNet), cross-entropy (SegFormer) | Fair comparison requires documenting distinct losses |
| Hardware | Single NVIDIA RTX 3090, 24 GB VRAM, Ryzen 7950X, 64GB RAM | RTX A2000 12GB requires measurement / smaller tiles or batches / accumulation |

Detailed hyperparameters: official supplement Table 2, printed p. 10. Hardware: main paper section 4 and supplement p. 10.

**Reported holdout semantic IoU:** UNet R34 **0.838**, UNet R50 **0.849**, SegFormer MiT-B0 **0.865**, MiT-B5 **0.876**. These metrics are **pretrained-reference comparisons on the authors' 10 cm OAM-TCD holdout**, not predicted achievable performance for CC BY-only scratch training or GSI ~0.6 m/px. Main paper Table 1, printed p. 7.

## What the original runtime figures do and do not imply

Supplement Table 4 lists UNet ResNet34 **33 hours** and SegFormer MiT-B0 **33 hours** as an *aggregate estimate for six runs* (five cross-validation runs plus holdout), on an RTX 3090 setup. The same table lists larger models taking longer. These numbers **cannot** be directly applied to a single scratch run on an RTX A2000 12GB. Note that the paper also describes largest individual model runs as around a day; avoid confusing per-run and multi-run totals.

## Interpretation

1. The user hypothesis “little difference in accuracy because 4608 CC BY images vs 5072 all images” is **plausible only when controlling other variables**. The number difference alone is not an estimate of accuracy change; the training subset distribution, license-specific biomes, oam_id source split and possible annotation bias must be checked.
2. The **bigger unmeasured gap** is *pretrained initialization vs random initialization*. The source indicates faster convergence from pretrained weights, but offers **no controlled final-IoU gap for training from scratch**.
3. An additional confounder is **different architectures**: MONAI BasicUNet and independent torchvision R34 U-Net aren't pixel-for-pixel equivalent to the published OAM-TCD versions; changing architecture and initialization simultaneously prevents attributing the gap to either one.
4. The ultimate operational criterion is GSI 0.5–0.6 m/px Japanese imagery, for which the original 10 cm benchmark is not a guarantee.
5. **Do not change the project’s scratch-training and CC BY-only baseline implicitly.** Independently rights-cleared pretrained weights *could* be researched as a separate future option, but their legal provenance would need verification and this is not an approved switch.

## Proposed controlled experiments, after G0 and P1 data QA

- **E0:** same CC BY-only training/validation groups, identical initialization protocol, deterministic seed, same architecture, compare 10 cm (native) and simulated 50–60 cm input; inspect whether learning works at all from scratch.
- **E1:** MONAI BasicUNet scratch vs independent R34-U-Net scratch with same split and GSD, common evaluation. Report val IoU/F1/precision/recall, crown-cover bias, boundary errors, elapsed training and peak VRAM. Report architecture difference explicitly, not as a pretraining effect.
- **E2 (only if justified):** extend scratch epochs and learning-rate schedule while holding split fixed; 75 epochs from OAM-TCD pretrained runs should **not** be treated as assured convergence for scratch.
- **Optional future targeted rights audit:** separately investigate truly permissive pretrained backbones, but **do not use checkpoints** in the baseline without new approval.

Acceptance evidence still missing: real CC BY dataset pixel QA, actual source-grouped train/validation splits, release-specific import/dependency audit, offline runnable models, company GPU timings, GSI independent ground truth. This research is *not* a G0 pass.

## Primary references

1. Veitch-Michaelis et al., **OAM-TCD: A globally diverse dataset of high-resolution tree cover maps**, NeurIPS 2024 Datasets and Benchmarks, §4 and Table 1: https://openreview.net/pdf/9c48222698818db2bebb7bfa7ee994f2785bbe6b.pdf
2. **OAM-TCD Appendix / Supplementary Material**, §2.1, Tables 2 and 4: https://papers.nips.cc/paper_files/paper/2024/file/58efdd77196fa8159062afa0408245da-Supplemental-Datasets_and_Benchmarks_Track.pdf
3. [Restor TCD project README](https://github.com/Restor-Foundation/tcd/blob/main/README.md), fetched blob `38756c5a64325cb2e1d0ed4314e9f71e7f62df3d`: currently released model weights are restricted; developers plan CC BY-only retraining and recommend alternate clean architectures for commercial settings.
4. [Restor SegFormer pipeline code](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/segformer.py), fetched blob `25be0be464a1d0bb1a2709cace606ef7370c9605`: uses `SegformerForSemanticSegmentation.from_pretrained` when loading configured published weights, but **this is model loading for use; it does not by itself prove the original training starting checkpoint**.

Research note: PDF screenshots were attempted in web verification but the service returned fetch failures; claims above were checked against extracted textual sections and linked code, not against a verified screenshot. 
