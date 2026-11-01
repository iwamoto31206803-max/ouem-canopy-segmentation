# PR #6 supplemental verification — source, dependency and smoke-test acceptance (v0.1)

Date: 2026-11-01. **Evidence-only audit; no G0 acceptance and no user-PC execution.**

## Verified from identified official release code and upstream documentation

### MONAI BasicUNet

- Official source release: `Project-MONAI/MONAI` tag **1.6.1**, `monai/networks/nets/basic_unet.py`, blob SHA `3b47fa0b030ada21210f85b72d1e4be0a76c45a1`. Header explicitly Apache-2.0; project LICENSE SHA `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.
- Official release `pyproject.toml` SHA `aa434b364fa375e1c54bd1aaf65ff8ad50562d6a`: `requires-python = >=3.10`, core deps `torch>=2.8.0`, `numpy>=1.24,<3.0`. Confirmed via source and [PyPI release page](https://pypi.org/project/monai/1.6.1/), but **not independently via installed wheel METADATA**.
- The class constructor defaults to **3D spatial**, so force 2D; also defaults to 2 output channels, so choose the desired binary formulation explicitly.
- **Minimal source-supported constructor**: `BasicUNet(spatial_dims=2, in_channels=3, out_channels=1)`. A single logit with `BCEWithLogitsLoss` or suitable binary focal criterion is one possible loss convention; do not apply a sigmoid twice when using logits loss.
- The constructor does not point to an OAM-TCD or ImageNet checkpoint. Confirm offline no-network instantiation in a later smoke test; no runtime execution claimed here.
- MONAI is a distinct U-Net variant and does **not preserve OAM-TCD SMP U-Net ResNet34 architecture**.

### torchvision ResNet34

- Official tagged `pytorch/vision` **v0.20.1** `torchvision/models/resnet.py`, blob SHA `83c0340cef74d9cb4c1dc92c38f4b3024be1f731`, has `def resnet34(*, weights=None, ...)` and distinct `ResNet34_Weights.IMAGENET1K_V1`.
- BSD-3-Clause repository LICENSE at same tag, blob `1edcf92c3317b90fedd187e2eaad101bd1c1efc5`. This is *code* license evidence, not individual checkpoint permission.
- Explicit `weights=None` avoids downloading ImageNet weights and preserves an auditable scratch initialization path; a custom U-Net decoder is needed for semantic segmentation.
- **v0.20.1 is evidence for the source API only, NOT a proposed pin with MONAI 1.6.1**: compatibility must be established from a mutually supported release matrix, then verified in the installed environment.

### Exact pretrained ResNet34 model identity and unresolved rights

- Official [torchvision ResNet34 API](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34) identifies `ResNet34_Weights.IMAGENET1K_V1` as DEFAULT.
- Official [torchvision source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html) identifies checkpoint `resnet34-b627a593.pth`; do not download into accepted training environment until permission is cleared.
- The general [torchvision pretrained model notice](https://docs.pytorch.org/vision/master/models.html) explicitly warns that pretrained models may carry independent terms based on training data.
- [ImageNet access agreement](https://image-net.org/accessagreement) explicitly permits access to ImageNet dataset only for **noncommercial research/educational purposes** (and also addresses for-profit researchers). This is a *dataset access restriction*; **do not automatically equate it to a legally established prohibition on all independently released downstream checkpoints**.
- No explicit per-checkpoint commercial grant or written rights clearance was located in the inspected official documentation. Therefore **checkpoint permissions = NOT VERIFIED**; neither automatically allowed nor definitively prohibited.

## Confirmed scope boundaries

| Audit item | Evidence | State |
|---|---|---|
| MONAI code Apache-2.0 | Tagged source and LICENSE | Source-level verified |
| MONAI Py3.11 eligibility | pyproject >=3.10 | Source-level verified |
| MONAI dependency floor | torch>=2.8, numpy>=1.24,<3 | Source-level verified |
| ResNet34 random init capability | torchvision tagged source weights=None | Source-level verified |
| ResNet34 ImageNet checkpoint identity | Official weights API/source | Verified identity, no use permission |
| All wheel METADATA and license/NOTICE files | Not downloaded or inspected here | **NOT VERIFIED** |
| Full transitive packages / CUDA runtime notices | Not installed or inspected | **NOT VERIFIED** |
| GEEKOM-A6 Windows CPU inference/backprop | No user device access | **NOT VERIFIED** |
| Company A2000 training / RTX T400 | No user device access | **NOT VERIFIED** |
| Empirical IoU/F1 / real OAM images | No experiment | **NOT VERIFIED** |
| License G0 acceptance | Requires all applicable gates | **NOT PASSED** |

## Proposed next implementation / validation PR (not part of #6)

1. Resolve compatible **PyTorch / torchvision / MONAI** install versions for Windows Python 3.11, capture pinned release artifacts/hashes and notices; **do not assume old torchvision 0.20.1 is a working combination with torch>=2.8**.
2. For **MONAI BasicUNet**, run CPU-only offline smoke with deterministic seed and small tensors (e.g. 128² and 256²); verify `(N,3,H,W)->(N,1,H,W)`, finite logits, loss, backward gradients, state save/load and parameter count.
3. Separately run `torchvision.models.resnet34(weights=None)`; verify no hub/cache/network checkpoint fetch. Keep custom decoder work in separate scoped PR, with its own tests and origin audit.
4. Use no real OAM-TCD images until P1 CC BY-only license-filtered dataset QA and source-grouped split are verified. Do not label synthetic smoke as real-data validation.
5. For eventual company RTX A2000 test, install Python only with company IT approval and company-approved package import route; enforce policy that GitHub pull is allowed while ChatGPT/Codex may not connect.

### Stop/go recommendation

For moving toward a runnable baseline, **MONAI BasicUNet scratch has a stronger simplicity/provenance argument** than a bespoke ResNet34 decoder. This is a *candidate recommendation*, not measured superiority. Keep A2 architecture as a next comparison once the dataset workflow is stable; pretrained ImageNet weights remain a deferred separately authorized route.

PR #6 can be reviewed/merged **as research-only documentation**, not as completion of G0.
