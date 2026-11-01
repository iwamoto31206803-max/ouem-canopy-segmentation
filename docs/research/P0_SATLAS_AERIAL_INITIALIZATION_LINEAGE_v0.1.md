# SatlasPretrain Aerial_SwinB_SI: second-level initialization provenance audit (v0.1)

**Date:** 2026-11-01. **Research only; commercial license G0 NOT PASSED; no model/weights downloaded or executed.** This **updates and qualifies** the positive-license discussion in `P0_PRETRAINED_FOUNDATION_WEIGHTS_CANDIDATES_v0.1.md`. Read these together.

## Critical newly verified finding

**The published SatlasPretrain aerial training configuration initializes the Swin-v2 Base backbone using torchvision ImageNet-1K pretrained weights before SatlasPretrain training.** The path, if this published configuration was used for the released artifact, is:

`torchvision Swin_V2_B_Weights.IMAGENET1K_V1` → SatlasPretrain aerial multi-task training → `Aerial_SwinB_SI` publicly released as **ODC-BY** → possible OUEM fine-tuning.

This changes the interpretation of a "commercial-clean alternative to ImageNet": **Satlas is NOT demonstrably an ImageNet-free weight lineage.** It **does** have a stronger, explicit publisher statement of license applied to the *final Satlas weight files*, but whether/how that interacts with original ImageNet checkpoint rights is a separate legal/provenance question.

## Evidence chain

| Step | Primary evidence | What is established | Remaining caveat |
|---|---|---|---|
| 1. Aerial model config | [Ai2 `configs/aerial/swinb_si_pretrain.txt`](https://github.com/allenai/satlas/blob/main/configs/aerial/swinb_si_pretrain.txt), blob `b2e38da43d7bb1a1a2baa4af192fecaeab2a2783` | `"Backbone":{"Name":"swin","Arch":"swin_v2_b","Pretrained":true}`, `NumChannels:3`, `Half:true`, crop ~416–512, resize 512 | Config in published repo, **not** a signed reproduction certificate of exact released weight-training run |
| 2. Implemented behavior | [Ai2 `satlas/model/model.py`](https://github.com/allenai/satlas/blob/main/satlas/model/model.py), blob `d3200b47f23a837316116b9c0f83255a3b88b34d` | In `SwinBackbone.__init__`, for `swin_v2_b` with `Pretrained:true`, calls `torchvision.models.swin_v2_b(weights=torchvision.models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1)` | Actual torchvision artifact version/published checkpoint hash and historical run record need tracing |
| 3. Stem replacement | Same code | Then replaces `self.backbone.features[0][0]` with new `Conv2d(num_channels,...,4,stride=4)`; so not *every* original checkpoint tensor necessarily survives the first construction | Does **not** eliminate ImageNet lineage from other layers |
| 4. Weight publication | [Ai2 model package README](https://github.com/allenai/satlaspretrain_models/blob/main/README.md) blob `864e7482557aac925a020614c313270c92286a77`; [Ai2 HF model card](https://huggingface.co/allenai/satlas-pretrain/blob/main/README.md) | Both explicitly say **model weights released under ODC-BY**; `Aerial_SwinB_SI` single-image aerial RGB ~0.5–2m/px | Exact checkpoint file content hash and any external legal obligations need separate assurance |
| 5. Data source rights | [Ai2 SatlasPretrain source and data listing](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md), blob `094ff770c2e2d8c4b6e2284bc9caf157d50a2087` | NAIP public-domain imagery among aerial inputs; annotations include mixed ODbL/CC-BY/ODC-BY sources | ODC-BY database grant explicitly does **not** automatically cover separately owned database contents; exact aerial-training use and downstream conditions require careful assessment |
| 6. Independent third-party catalog | [TorchGeo models docs, v0.8.0](https://docs.torchgeo.org/en/v0.8.0/api/models.html) and [TorchGeo Swin source](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/swin.py) | TorchGeo independently catalogs `NAIP_RGB_SI_SATLAS` as **ODC-BY**, `swin_v2_b`, 3 bands, and pins a replicated weight URI/revision | This corroborates label and technical interoperability, **not** a legal opinion or independent chain-of-title validation |

## License interpretation — carefully separated

- **Positive evidence:** Ai2 expressly grants its final released pretrained model *weights* under **ODC-BY** and links license terms that allow commercial use with attribution. The Apache-2.0 library and public-domain NAIP imagery are further positives.
- **Negative/uncertain evidence:** the training config and source reveal **prior ImageNet-1K weights** through torchvision. ImageNet dataset access agreements include noncommercial use restrictions. But dataset access restrictions are **not a definitive legal proof** that a model checkpoint further fine-tuned with other data cannot be commercially used; nor is Ai2's ODC-BY declaration necessarily sufficient to waive any independent third-party rights.
- **Therefore:** route Satlas is **promising with an explicit published commercial-use weight license**, but **NOT YET independently cleared for production**. *Do not* claim it is unequivocally commercial-clean, or conversely categorically banned by ImageNet.
- **Practical gate:** clarify with Ai2 publisher (and, if warranted, underlying torchvision/ImageNet rights holders or company legal team) whether the stated ODC-BY on `Aerial_SwinB_SI` covers **commercial fine-tuning, deployment, redistribution**, given the initial torchvision ImageNet weight provenance. Also ask whether the public config matches the precise released weight and what attribution notices they require. Seek provenance of individual bundled checkpoint revision / artifact hash.
- **Scope choice:** user has not authorized contacting Ai2 or downloading pretrained weights. This document **does not** send an inquiry or change project baseline.

## Architecture and resource observations

- Satlas model is **Swin-v2 Base** RGB, not ResNet34 / SegFormer / MONAI BasicUNet. Model size/training memory will be materially different; no measured A2000 12GB feasibility.
- Official model loader package `setup.py` at GitHub blob `d14a70a86b3d4ac743f120bd05e4a733828a16e7` states `python_requires>=3.9`, `torch>=2.1.0`, `torchvision>=0.16.0`, `requests`, `matplotlib`, package version `0.3.1`. These are *lower bounds*, not a compatible locked environment.
- Ai2 config `BatchSize=32`, `Half=true`, input resized to 512: from original upstream large-scale GPU training, **not** a feasible default for single RTX A2000 12GB without actual benchmark.
- The package automatically downloads checkpoint by model ID. In later approved tests, pin URL/revision/hash, cache and verify file before GPU use. **PyTorch `.pth` loading can invoke pickle**: treat downloaded binaries as trusted-source inputs and prefer a safe `weights_only=True` loader where supported; audit package's loader behavior before opening weights.
- A single image GSI orthophoto aligns with `Aerial_SwinB_SI`; multi-image model not assumed interchangeable.

## Updated priority and stopping rules

**Research priority:** investigate Ai2's exact weight grant + pretraining lineage first; next test whether **Swin-v2 Base with frozen backbone** or modest patch sizes fits RTX A2000, without asserting full fine-tuning fits. **Engineering fallback:** scratch MONAI BasicUNet remains clear starting point, but not a demonstrated accuracy-equivalent substitute.

**Decisive status table:**

| Question | Status |
|---|---|
| Satlas final weights have Ai2-published ODC-BY designation? | **YES — explicit first-party documentation** |
| ODC-BY generally permits commercial use? | **YES for rights granted by licensor, with required attribution** |
| Satlas aerial upstream ImageNet initial weights? | **YES in published config + implementation; exact release lineage still needs confirmation** |
| Does Ai2 license settle all third-party weight/data questions? | **NOT VERIFIED** |
| Can OUEM adopt Satlas for commercial training today? | **NOT CLEARED** |
| Does A2000 12GB support a usable Swin-v2 Base training scheme? | **NOT BENCHMARKED** |

### Follow-on work after research PR acceptance

1. Document a bounded rights-holder clarification request (only send with user approval).
2. If permitted, create independent checkpoint inventory: model ID, source revision, SHA256, LICENSE/NOTICE, data provenance.
3. Independently pin Python 3.11 Windows-compatible torch/vision / Satlas loader stack; synthetic CPU forward/frozen-feature smoke. **No company environment access or installation without relevant approval.**
4. Small-scale controlled evaluation against scratch baseline, then full 0.5–0.6m GSI adaptation as previously staged.
