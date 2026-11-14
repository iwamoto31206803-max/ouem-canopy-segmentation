# R2a-1 CPU multi-image training core

Issue #66 builds on PR #65 at `9e91b0e196a2d47d4474ca886976189e4c19f27b`.
`scripts/r2_train.py` supplies R1a-backed patches and a small CPU training API.
The current verification uses synthetic R1a PASS fixtures and tiny models.

## Entry points

`R1Data(root, manifest, expected_dataset_id, expected_manifest_sha256)` checks the
caller-pinned file hash and ID, runs unchanged R1a full `preflight()`, and retains
its loading context. Every patch uses unchanged `load_sample()` to recheck the
selected four products. A HOLD/FAIL cannot create the context. No historical
smoke acceptance is inherited. Inputs remain read-only.

`make_plan(data, seed=0, patch_hw=(224,224), steps=20, val_ids=None,
val_origin=(0,0), lr=1e-4, weight_decay=.01)` returns explicit configuration,
dataset identity and complete Train/Val schedules. Any positive step budget and
Train count are supported. Sizes may differ. Train IDs are sorted then shuffled
once per cycle with a local seeded RNG, with one appearance per cycle. Crop
positions follow actual sizes. Val defaults to all Val IDs in sorted order at a
fixed origin, independent of seed. The caller can specify a Val subset/origin.
Oversized patches fail before output creation. There is no padding or upscale.

All four products use the same crop. RGB is float32/255, target is long with
classes 0/1, and valid_fraction must be all one. Uniform teachers are preserved.

`train(data, plan, model_factory, fresh_output, stop_after=None, resume=None,
resume_sha256=None)` constructs a seeded CPU FP32 model and AdamW, trains up to
the explicit stop/budget, and saves checkpoint + JSON result. The factory returns
a model with the established Satlas call contract
`probabilities, ce_loss = model(images, target)`. The tiny test model uses normal
cross entropy. No new loss, official model construction or weight download is
implemented. Validation patches can be loaded through `data.patch(item, 'val')`.
Validation metrics and a production/GPU entrypoint are future work.

```python
from r2_train import R1Data, make_plan, train

data = R1Data(root, manifest, expected_dataset_id, expected_manifest_sha256)
plan = make_plan(data, patch_hw=(32, 48), steps=6, seed=0)
first = train(data, plan, model_factory, first_output, stop_after=2)
continued = train(data, plan, model_factory, next_output,
                  resume=first_output / 'checkpoint.pth',
                  resume_sha256=first['checkpoint_sha256'])
```

## Checkpoint and failure behavior

Existing M1-6 `save_state()` / hash-pinned `safe_load(weights_only=True)` and G2
JSON writing are reused. The checkpoint contains the full plan, dataset ID/file
SHA, completed steps/history, model/AdamW state and CPU Torch/Python/NumPy RNGs.
Resume checks plan/data/configuration/history and optimizer settings, strict-loads
the model and restores RNGs after construction. Its next step comes from the
stored completion count. The model factory must preserve architecture and forward
semantics. A CPU continuation equality test covers a stochastic tiny model in the
same environment. Cross-version, GPU and official Satlas bitwise equality remain
unverified. The new API allows deliberate branching into different fresh outputs.

Each segment requires a new directory outside the input dataset and source
checkpoint directory. Step JSONs and a terminal result are retained. Checkpoint
bytes are written exclusively to a partial file before exclusive hard-link
publication. Write failure retains the partial and FAIL result, preserving the
source checkpoint. There is no automatic retry or power-loss durability claim.
If writing the failure report also fails, the API raises and returns no PASS.
Checkpointing occurs at the segment end. A failed segment can resume from its
last previously completed segment using a new output directory.

## CPU verification

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_r2_train.py tests/test_r1_dataset.py tests/test_m1_7_g2_train.py
.\.venv\Scripts\python.exe -B -m pytest -q
```

Tests create explicit synthetic rights/review PASS assertions in temporary
manifests, variable image sizes and a tiny stochastic CE model. These assertions
apply only to generated test pixels. Existing real candidates remain Formal R1
HOLD, G3/G4 remain PENDING and R1d `REAL_EXECUTION_DISABLED` is unchanged.
G0–G3 implementations, source data, checkpoints and acceptance contracts are
unchanged. No actual OAM/GSI/Test/GT54 pixels or GPU are used in this work.

On 2026-11-14, the CPU full regression suite completed with **563 passed,
1 skipped, 4,066 warnings in 224.50 seconds**, including the existing R1a/G2
regressions. The skip is the existing Windows symlink-privilege case. Synthetic
continuation matched model, AdamW, all three RNG states and loss history, and
the focused rerun also checked equal restored prediction tensors. This result
does not establish official Satlas/GPU performance or actual image quality.
