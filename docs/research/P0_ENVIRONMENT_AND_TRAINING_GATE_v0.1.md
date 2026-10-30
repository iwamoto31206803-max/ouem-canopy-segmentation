# P0 training-environment and version gate (v0.1)

Date: 2026-10-31. **Research-only requirements; no code, model download or training.**

## User-confirmed constraints

- Commercially usable model construction is a **hard selection gate**.
- **Train from random initialization** (no OAM-TCD published noncommercial pretrained model weights); restrict training imagery and annotations to a provenance-checked **CC BY 4.0 subset**. This is an accepted cost, not an objection to the route. Revisit a pretrained backbone only as a separately documented/cleared optional alternative; do not make it the baseline.
- **Development machine:** GEEKOM-A6 Ryzen 9 6900HX, 32 GB RAM, Windows, Codex/GitHub. Use for code, reproducibility, CPU-only smoke tests, bounded sample QA and tests.
- **GPU training machine:** company PC with NVIDIA RTX A2000. Do **not** reject a model merely because GEEKOM-A6 cannot train it at useful speed. Exact VRAM, driver/CUDA and package versions need verification on the actual GPU machine. Company policy prohibits direct ChatGPT/Codex connection; move reviewed code via GitHub pull and handle datasets appropriately.
- No claim that RTX A2000 will fit any arbitrary batch size or model; establish VRAM and then use tile size/batch/AMP/gradient accumulation as appropriate.

## Newly inspected upstream compatibility evidence

1. [MONAI development-branch pyproject.toml](https://github.com/Project-MONAI/MONAI/blob/dev/pyproject.toml), blob `aa434b364fa375e1c54bd1aaf65ff8ad50562d6a`, declares Python >=3.10 and **torch>=2.8.0**, numpy>=1.24,<3.0 for this development branch. This **does not** establish compatibility with an older torch build on the company machine; choose a release compatible with the actual installed stack, or use a separate approved environment. A development branch is NOT a release pin.
2. [MONAI BasicUNet implementation](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/basic_unet.py), blob `3b47fa0b030ada21210f85b72d1e4be0a76c45a1`, supports `spatial_dims=2` RGB and single-channel output when explicitly configured; source is Apache-2.0. This is a separate architecture from ResNet34 encoder U-Net.
3. [torchvision ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py), blob `47067ec83175a97cc6f6a8721b342128a434d440`, provides the ResNet34 code path; `weights=None` should be verified in a pinned release before asserting an offline clean start.
4. [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) is BSD-3-Clause, but package and binary distribution notices must still be reviewed. Python/PyTorch/CUDA binaries and transitive dependencies may have additional notices.

## Environment-aware staged decision

**Stage A — P0 legal/source/version gate:** select a *pinned* released implementation and dependency set; confirm model-source and artifact licenses, initialization without restricted weights, notices. Current state: not passed.

**Stage B — offline CPU smoke (GEEKOM):** in isolated virtual environment instantiate from random weights, monkeypatch/deny network, forward small RGB tensors, check logits shape, determinism and dependency manifest; no full training. No execution yet.

**Stage C — bounded data QA (GEEKOM):** download approved CC BY-only OAM-TCD records, audit attribution and license, inspect real masks and grouped split; never mix NC/SA source records. No pixel-level acceptance yet.

**Stage D — GPU feasibility (company):** record VRAM and CUDA versions; run identical reproducible small-batch model comparison on the A2000. Choose tiles/batch/AMP/accumulation based on measured peak VRAM, throughput and quality, not guessed device specs. Ensure GPU workflows obey company policy.

**Stage E — quantitative GSD domain adaptation:** training and validation from CC BY-only records, grouped by source; assess simulated 0.5–0.6 m/px and Japan GSI image transfer, crown area bias and boundary errors. This is not yet approved implementation.

## Deep Research decision

A comprehensive literature/license provenance review could be useful when comparing current commercial-clean architecture families, alternative pretraining data and nationwide remote-sensing transfer learning. **It is optional, not a prerequisite for the narrowly scoped primary-source software/version checks in Stage A.** Use it for a defined question, e.g. “What commercially eligible initialization and architecture strategies exist for 10 cm aerial canopy segmentation transferred to 0.6 m GSI imagery, with source-backed license matrix and A2000-feasible training evidence?” No Deep Research execution is claimed here.

**No formal architecture approval, no G0 PASS, no PR merge.**
