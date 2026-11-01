# PR #6: Commercial-clean model route and pretrained-weight license gate (v0.1)

Date: 2026-11-01. **Research-only decision note**; no trained model or checkpoint is approved by this document. Continuation of merged PR #4 architecture/source trace and PR #5 OAM-TCD paper review.

## Executive finding

**A. Random initialisation with audited code and OAM-TCD CC BY 4.0-only training data remains the baseline with the fewest pretrained-checkpoint license questions.**

**B. An ImageNet-pretrained ResNet34 encoder is a technically plausible route, but its eligibility for commercial work is NOT YET VERIFIED.** The torchvision library is BSD-3-Clause; **that is not a sufficient license grant for its hosted ImageNet-pretrained weight file**. Torchvision explicitly warns that pretrained models may carry licenses or usage terms derived from their training datasets. ImageNet's own dataset access agreement restricts dataset use to non-commercial research/education. It does **not by itself settle whether such restrictions legally attach to an independently distributed derivative weight file**. Further rights-holder clarification is necessary before commercial approval. Do not label B as outright illegal or already cleared.

## Precise evidence and confidence

| Component | Official source / identifying artifact | Finding | Status |
|---|---|---|---|
| torchvision ResNet34 implementation | `pytorch/vision` **v0.20.1**, `torchvision/models/resnet.py`, SHA `83c0340cef74d9cb4c1dc92c38f4b3024be1f731` | `resnet34(weights=None)` constructs an unpretrained backbone; `ResNet34_Weights.IMAGENET1K_V1` points to `resnet34-b627a593.pth` | **Source confirmed**, runtime not tested |
| torchvision project license | v0.20.1 LICENSE, SHA `1edcf92c3317b90fedd187e2eaad101bd1c1efc5` | BSD-3-Clause, subject to notice/redistribution conditions | **Code license source confirmed**; full wheels/dependency notices not audited |
| torchvision official pretraining weight specification | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34 ; https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html | `IMAGENET1K_V1` only standard ResNet34 weights; `DEFAULT` resolves to this; training/accuracy recipe listed, but **no independently verified explicit commercial weight grant** from page | **Identity confirmed; usage permission unresolved** |
| torchvision blanket weight caveat | https://docs.pytorch.org/vision/master/models.html | Docs warn: pretrained models may have licenses/terms from training datasets and user is responsible for checking permission | **Direct official warning** |
| ImageNet dataset license/terms | https://image-net.org/accessagreement ; https://image-net.org/download-images/login | Dataset made available for non-commercial research and educational purposes; image copyrights aren't owned outright by ImageNet | **Data restrictions confirmed; derivative-weight implications unresolved** |
| MONAI BasicUNet | `Project-MONAI/MONAI` **1.6.1**, `monai/networks/nets/basic_unet.py`, SHA `3b47fa0b030ada21210f85b72d1e4be0a76c45a1`; LICENSE SHA `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | Apache-2.0 implementation, standard constructor `BasicUNet(spatial_dims=2,in_channels=3,out_channels=1)`; no external pretrained weight is needed for this route | **Code-source evidence**, full dependency/wheel/runtime audit still pending |
| PyTorch core | https://github.com/pytorch/pytorch/blob/main/LICENSE | Separate BSD-style licensing; inspect locked binary / runtime notices | **Broad project license confirmed** |
| SMP U-Net ResNet34 | Prior PR #4 `P0_UNET_R34_SOURCE_TRACE_v0.1.md` | `encoder_weights=None` avoids checkpoint download but package import graph includes NVIDIA-licensed SegFormer material | **Implementation provenance gate unresolved**; do not prematurely say mere import is itself a legal violation |

**Crucial caveat:** ImageNet's noncommercial *data* terms do not automatically give an unambiguous legal answer on *checkpoint distribution or downstream commercial exploitation*. An explicit license notice for the exact weight artifact, dataset-rights analysis, or written confirmation is required. The hypothetical statement “torchvision BSD license therefore weights BSD” is **not justified**.

## Routes compared for OUEM

### Route A1 — MONAI BasicUNet + random initialization (simplest provenance)

- Full model trains from scratch on audited CC BY 4.0 canopy imagery/labels.
- Only MONAI BasicUNet implementation, PyTorch and runtime/binary dependencies require clearing; MONAI source is Apache-2.0.
- Architecture differs from OAM-TCD's published **SMP U-Net with ResNet34/50 encoder**, so performance cannot be projected from original published U-Net IoU.
- First tests: no-download instantiation, 3-channel 2D input, 1-channel logits, arbitrary input size/padding, gradients, serialization, deterministic seed, memory footprint, CPU smoke.

### Route A2 — torchvision ResNet34(weights=None) + independently authored U-Net decoder

- Preserves ResNet34 encoder architecture without relying on SMP's eager imported third-party material or ResNet34 ImageNet weights.
- Decoder and skip connections need independent implementation and tests; ensure no copying from restricted source and record provenance.
- Need test encoder stage output scales/channels; input dimensions and padding; training behavior, runtime and license notices.
- **Not equivalent to OAM-TCD published SMP U-Net** unless matching decoder and losses verified.

### Route B — torchvision ResNet34 pretrained `IMAGENET1K_V1` + independently authored decoder

- Official checkpoint: `https://download.pytorch.org/models/resnet34-b627a593.pth` (identifier only; **NOT downloaded or approved**).
- Source code/layers same as route A2, but initial encoder weights trained on ImageNet.
- Expected potential advantage: accelerated convergence / improved sample efficiency (hypothesis consistent with OAM-TCD paper but **not demonstrated on CC BY-only GSI**).
- **Commercial checkpoint gate open** due exact-weight permission and ImageNet access terms. **Do not put downloaded pretrained checkpoint into any commercial-output pipeline without rights clearance.**
- Would require a *separate user decision* to change the established scratch baseline; no silent switch.

## Recommended practical decision

1. **Proceed with a cleared scratch route for engineering**, initially **MONAI BasicUNet** for smallest original-code burden; retain **torchvision ResNet34 random + custom decoder** as architecture-comparable fallback/candidate. This is a *research recommendation*, **not final selection or G0 passed**.
2. **Keep pretrained ResNet34 as a deferred candidate** pending specific explicit right to use checkpoint commercially. If that clarity is not forthcoming, do not block useful scratch experiments.
3. Source-pin production packages, collect all license notices for installed distribution files, inspect import graph, run offline functional tests, and only then decide **G0**. Actual wheel METADATA and binary/third-party obligations are not yet validated.
4. Dataset P1 quality/split gate is independent. License-filter and group by `oam_id`; verify masks, iscrowd/void, imagery and biome composition, attribution.
5. Never compare GSI 0.6m performance or CPU/GPU time via unmeasured estimates from native OAM-TCD 10cm data.

## Precise next acceptance checklist (separate from this research PR)

- [ ] Package versions and Python 3.11 compatibility pinned to a **mutually supported** torch/torchvision/MONAI combination; no blind use of historical torchvision v0.20.1 as an environment recommendation.
- [ ] Exact PyPI/wheel metadata, transitive notices and license files checked (MONAI/PyTorch/torchvision / CUDA binaries as appropriate).
- [ ] Offline environment with no checkpoint download (explicit `weights=None`; MONAI BasicUNet creates fresh parameters).
- [ ] CPU forward/backward tests at representative patch shapes and output dimensions.
- [ ] Model & initialization records include seed, source tags/SHAs and checkpoint hash if applicable.
- [ ] If pursuing Route B: written or explicit official commercial authorization for the **exact checkpoint**, plus assessment of upstream image dataset restrictions.
- [ ] Separate real CC BY-only pixel/mask/source attribution QA and oam_id grouped split.

## Decisions NOT taken

- **No approval for using pretrained weights**, including torchvision ImageNet weights.
- **No conclusion that ImageNet-pretrained weights are necessarily illegal commercially**; legal rights unresolved.
- **No equivalence claim for MONAI BasicUNet and OAM-TCD's U-Net ResNet34.**
- **No G0 pass, model implementation, training, local test, company PC setup or checkpoint download.**

## Reference URLs (official)

- torchvision models and weight caveat: https://docs.pytorch.org/vision/master/models.html
- ResNet34 official weight API: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34
- torchvision pinned-source file: https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py
- torchvision pinned license: https://github.com/pytorch/vision/blob/v0.20.1/LICENSE
- ImageNet Access Agreement: https://image-net.org/accessagreement
- ImageNet official data page: https://image-net.org/download-images/login
- MONAI BasicUNet pinned source: https://github.com/Project-MONAI/MONAI/blob/1.6.1/monai/networks/nets/basic_unet.py
- MONAI pinned license: https://github.com/Project-MONAI/MONAI/blob/1.6.1/LICENSE
- MONAI BasicUNet docs: https://monai.readthedocs.io/en/latest/networks.html
