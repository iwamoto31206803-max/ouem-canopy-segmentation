# P0 official release and dependency gate (v0.1)

Date: 2026-10-31. Research only. This document supplements P0_ENVIRONMENT_AND_TRAINING_GATE_v0.1.md; no model implementation or training.

## Confirmed official distribution versions

| Package | Latest PyPI release observed | Release date | Core license | Distribution |
|---|---|---|---|---|
| MONAI | **1.6.1** | 2026-09-27 | Apache-2.0 | pure Python wheel and source |
| segmentation-models-pytorch (SMP) | **0.6.0** | 2026-10-26 | MIT package-level, **component exceptions** | pure Python wheel and source |

Official sources: https://pypi.org/project/monai/ ; https://pypi.org/project/segmentation-models-pytorch/ . SMP's PyPI page **explicitly warns** of file-specific licenses; the MIT metadata is insufficient to clear the bundled SegFormer encoder. The SMP 0.6.0 wheel and sdist have PyPI attestations, but publication provenance is not a commercial license grant. MONAI 1.6.1's PyPI files have no attestations shown; this is not itself evidence of unsafe code.

## Dependency distinction

MONAI's official installation guidance states its **core** direct dependencies are NumPy and PyTorch; other features are optional. https://monai.readthedocs.io/en/stable/installation.html . Its development-branch `pyproject.toml` inspected earlier declared `torch>=2.8.0`, but this **must not be silently assigned to the pinned 1.6.1 wheel** without inspecting its actual metadata.

The release numbers alone are not a reproducible environment lock. Before a runtime test:
1. Extract the exact `Requires-Dist` and `Requires-Python` from **MONAI 1.6.1 wheel** (or PyPI JSON for that release).
2. Extract notices and included files from the exact MONAI wheel, especially package init/import behavior; verify only intended dependencies and license files.
3. Verify installed Python, PyTorch, torchvision, CUDA, GPU driver, VRAM on the **company** GPU PC. Do not infer GPU capability from the GEEKOM-A6 CPU machine.
4. Record hashes of selected wheels, plus full `pip freeze`/dependency manifest.
5. In a clean GEEKOM-A6 venv, offline smoke-test MONAI BasicUNet 2D RGB to one logit map. No model weights or training data should be fetched.
6. For a ResNet34 comparator, verify `torchvision.models.resnet34(weights=None)` in a pinned torchvision release, and a permissively licensed independent decoder.

## Decision

**Shortlist for G0:** (a) MONAI 1.6.1 BasicUNet 2D from random initialization; (b) pinned torchvision ResNet34 with independently licensed U-Net decoder. SMP 0.6.0 is a documented comparator but not the default commercial-clean implementation due to bundled/imported file-specific noncommercial SegFormer code.

**Unresolved:** exact release wheel dependency/license contents, compatibility with actual company environment, no-network runtime smoke, data provenance and accuracy. G0 NOT PASSED. No implementation or training approved by this note.
