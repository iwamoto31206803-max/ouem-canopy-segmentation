# P0 — Permissively licensed U-Net alternatives (source audit v0.1)

Date: 2026-10-31. Research only; no installed package test, no completed dependency license audit, no model selection/approval.

## Discovery: a separate ready-made alternative to SMP

**MONAI BasicUNet** implements spatial_dims=2, in_channels=3, out_channels=1, configurable feature widths. It is an existing CNN U-Net, not a ResNet34 encoder-decoder. Source file is explicitly Apache-2.0, and its package declares Apache-2.0. This significantly reduces need to invent a new U-Net but does not automatically clear every runtime/transitive dependency and distribution form.

Sources:
- https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/basic_unet.py (inspected blob SHA `3b47fa0b030ada21210f85b72d1e4be0a76c45a1`; defaults include `spatial_dims=3`, `in_channels=1`, `out_channels=2`: all must be overridden for RGB binary canopy)
- https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE (inspected blob SHA `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`)
- https://github.com/Project-MONAI/MONAI/blob/dev/pyproject.toml (official Apache-2.0 package declaration and Python >=3.10; PyTorch and NumPy required).

**Source-level candidate shape** (not executed or adopted):
```python
from monai.networks.nets import BasicUNet
model = BasicUNet(spatial_dims=2, in_channels=3, out_channels=1)
```
No pretrained checkpoint is referenced by this constructor in the inspected BasicUNet source. Verify package import side effects and actual no-network constructor via runtime smoke test before G0 clearance. Use output *logits*, with sigmoid/calibration outside model.

Important: MONAI BasicUNet is **not** equivalent to Restor U-Net R34; parameter count, receptive field, normalization, feature width and performance will differ. No claims of superiority to R34.

## Independent ResNet34 route

Build **only the U-Net decoder/skip adapters needed** with `torch.nn`, feed it a `torchvision.models.resnet34(weights=None)` encoder. Avoid SMP entirely; inherit only permissively licensed torch/torchvision components and *newly written* decoder (don't copy restricted source). Input shape, skip feature selection, strides, padding, normalization and final output logits need clear tests. A thin bespoke decoder is the price of preserving ResNet34 without SMP package import ambiguity.

Sources:
- https://github.com/pytorch/vision/blob/main/LICENSE (BSD-3-Clause, inspected blob SHA `1edcf92c3317b90fedd187e2eaad101bd1c1efc5`).
- https://github.com/pytorch/pytorch/blob/main/LICENSE (main PyTorch BSD-style license, inspected blob SHA `c23172f7aff0254e4f0f163fb2e6e355cbaec5f4`).
- https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py (ResNet34 weights argument, no pretrained checkpoint for `weights=None`).
- https://github.com/pytorch/pytorch/blob/main/pyproject.toml (many separately licensed components/dependencies; official metadata describes mixed licenses beyond core BSD; packaging-specific notices remain).

## Comparison, evidence only

| Candidate | Existing source | Encoder | Restricted third-party checkpoint path | Status |
|---|---|---|---|---|
| M1: MONAI BasicUNet 2D | Apache-2.0 file | MONAI's own conv blocks | None shown in inspected constructor | **Promising candidate**; runtime import/deps and workload tests needed |
| M2: torch/torchvision ResNet34 + independently implemented decoder | torchvision BSD-3; custom code license must be chosen | ResNet34 random initialization | `weights=None` deliberately avoids pretrained download | **Promising construction path**, requires own model code and tests |
| M3: SMP U-Net R34 random | SMP MIT core but eager imports NVIDIA-licensed module | ResNet34 random | Weights not loaded if explicit None | **Licensing ambiguity** of complete packaged code/import |
| M4: HF SegFormer clean implementation | Apache file header; independence/provenance not settled | SegFormer no NVIDIA weights | Unverified complete independence | **Further IP assessment needed** |

No weight lineage exists yet; nothing has been trained. “Promising” never equals an approved commercial-use license status.

## More important than the library name: acceptance tests

- Pin **actual released** packages, not moving `main`/`dev`, and obtain notices/licenses of distributed wheel incl. transitives.
- Verify model constructor and first forward pass in a *network-disabled* environment, with package import list/no attempted HF/model download.
- Verify arbitrary test tensors with shapes including H/W not multiples of 16: either explicit model limitations or documented pad/crop behavior and output alignment.
- Verify all trainable parameters start from a recorded seed (not pretrained), reproducible initialization and model config.
- Compare exact model parameters, CPU inference time and peak memory under a fair test image, rather than infer from simplicity.
- Decide attribution/redistribution process for both MONAI and PyTorch dependencies.
- Assess whether MONAI's broad packaging is worth the extra deps versus a small independently implemented decoder and torchvision.

## Decision and next investigation

Keep M1 **MONAI BasicUNet** and M2 **torchvision ResNet34+clean U-Net decoder** as two primary candidates for G0. Both are source-level plausible for a commercial path, **neither is yet G0 PASS**. Do not choose based on claimed accuracy with no experiments. Initial next step: exact version/release + dependency/NOTICE and packaging audit, then a tiny offline forward/no-weight-download smoke check. After G0, obtain only bounded CC BY training records for GT QA.

**Explicit non-goal:** do not modify user model code, start full training, use restricted pretrained checkpoints or merge this research Draft automatically.
