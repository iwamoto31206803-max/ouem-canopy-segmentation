# P0 — release-tag verification and GPU preflight (v0.1)

Date: 2026-10-31. Primary-source verification; no runtime or package wheel inspection, no model training.

## Confirmed release source: MONAI 1.6.1

The GitHub tag `1.6.1` resolves and its `pyproject.toml` was fetched through the connected GitHub source:
- https://github.com/Project-MONAI/MONAI/blob/1.6.1/pyproject.toml — blob `aa434b364fa375e1c54bd1aaf65ff8ad50562d6a`
- `requires-python = ">=3.10"`
- core runtime dependencies `torch>=2.8.0`, `numpy>=1.24,<3.0`
- license declaration Apache License 2.0.
- build-system also requires torch>=2.8.0 and packaging/build tools; **build requirements are distinct from runtime requirements**.
- `monai/networks/nets/basic_unet.py` at the same tag, blob `3b47fa0b030ada21210f85b72d1e4be0a76c45a1`, implements configurable 2D BasicUNet and Apache-2.0 file header.
- `LICENSE` at tag, blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`, Apache-2.0.

This resolves the earlier uncertainty about whether development-branch torch>=2.8 applies to the 1.6.1 **tagged source**. It does. The distributed wheel's actual `METADATA` still needs independent confirmation; tag source does not prove exact wheel contents.

PyPI monai 1.6.1 wheel SHA256: `dbd3b1dd309e752e0477ed73fde03b92f80ac74bfb8a8c4260f018b5d6ab2b0b`; sdist SHA256: `e24ae4bb944f1779b7a6445acc4ff2efd9cc504d05dffe2498c49af117350136`.
Source: https://pypi.org/project/monai/1.6.1/

## Independent ResNet34 source baseline

Official torchvision `v0.20.1` tag exposes `resnet34(*, weights=None, ...)` with default None in `torchvision/models/resnet.py`, blob `83c0340cef74d9cb4c1dc92c38f4b3024be1f731`. This is source-level support for the no-pretrained-checkpoint construction. **Do not choose v0.20.1 as the final package pin until compatibility with the company environment is established.**
Source: https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py

## Company PC preflight: commands for an authorized local operator

In PowerShell on the company machine (no ChatGPT/Codex connection; do not share identifying system paths or private data):

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
py -3.11 -c "import sys; print(sys.version)"
py -3.11 -c "import torch; print('torch', torch.__version__); print('cuda_runtime', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
py -3.11 -c "import torchvision; print('torchvision', torchvision.__version__)"
```

These commands are **not guaranteed** to work if `py -3.11` or packages are not installed; failures are information, not instructions to install or change the company environment. Do not upgrade company torch or drivers until constraints and company approval are clear.

## Gate status

- Release tag / core dependency version: **source verified**.
- PyPI wheel exact METADATA / included licenses / package import trace: **NOT VERIFIED**.
- Company RTX A2000 driver, VRAM, installed torch/torchvision: **NOT VERIFIED**.
- Offline no-network BasicUNet forward and model-weight absence: **NOT VERIFIED**.
- Training data provenance / pixel QA: **NOT VERIFIED**.
- Model selection / commercial G0: **NOT PASSED**.

Recommendation: gather company preflight facts, verify exact wheel metadata, then select compatible environment and do CPU offline smoke. No training or implementation in this step.
