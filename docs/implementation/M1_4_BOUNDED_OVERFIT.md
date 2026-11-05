# M1-4 first optimizer step and bounded overfit

## Acceptance results

Experiment M1-4-CPU-20steps-seed0, observed 2026-11-05. OPTIMIZER_STEP_PASS, OVERFIT_DIAGNOSTIC=IMPROVED, CHECKPOINT_RESUME_PASS and RESOURCE_MEASURED. The [training result](M1_4_TRAINING_RESULT.json) records all twenty per-step losses, norms, update comparisons and memory/timing snapshots. Its historical final status indicates reload pending at that process's exit. The subsequent [separate-process reload result](M1_4_RELOAD_RESULT.json) completes the state-restoration acceptance.

The existing Train 661 top-left 256 x 256 and Val 1972 top-left 224 x 224 were loaded through M1-3A hash/grid/range/validity validation. Assigned oam_id/split and dataset revision d97d4da0ebbb6e249ae95ac5e19656babd972eb2 were preserved. License/attribution references are in the result and external STAC records. The official initializer SHA-256 was checked before safe weights_only CPU loading. Backbone/FPN loaded strictly, with fresh Upsample and two-class Head.

## Configuration and first-step gate

CPU FP32, four threads, seed 0, batch 1, AdamW lr=1e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8. All parameters were trainable. Exactly twenty optimizer steps used the same Train patch. No scheduler, augmentation, padding or ignore labels were used. Data and initializer were neither fetched nor transformed again.

Step 1 official CE was 0.692602456, gradient norm 0.04657543. Finite nonzero gradients were verified in each component, and active representative tensors were cloned before optimizer.step and compared afterward.

| Component | Representative | Maximum absolute first-step change |
| --- | --- | --- |
| Backbone | backbone.features.0.0.weight | 0.0001001954 |
| FPN | fpn.inner_blocks.0.0.weight | 0.0001001507 |
| Upsample | layers.0.0.weight | 0.0001000222 |
| Head | layers.0.0.weight | 0.0001000259 |

All updated model parameters were finite. AdamW created state for 419 parameters and state tensors were finite. Zero/absent-gradient parameters were excluded from representative selection. Each later step repeated finite loss/gradient/parameter/state checks and representative update comparisons. Only after step 1 passed did step 2 proceed. An 8 GiB private-commit guard samples stages and stops at the next check if exceeded. It cannot prevent an allocation spike or OS-level termination between samples. Ordinary exceptions yield FAIL_STOPPED and partial result evidence; an OS kill may prevent JSON emission.

## Comparable evaluation before and after

Both initial and final evaluations used model.eval() and torch.no_grad(), on the same respective crops. Classification uses argmax probabilities. Undefined metric denominators produce JSON null. Confusion counts TP/FP/FN are included in the actual result; zero denominators are never silently treated as perfect scores.

| Crop | CE before / after | Tree IoU before / after | F1 before / after | Precision before / after | Recall before / after |
| --- | --- | --- | --- | --- | --- |
| Train 661, 256 x 256 | 0.692601 / 0.323572 | 0.529561 / 0.845756 | 0.692435 / 0.916433 | 0.529764 / 0.886281 | 0.999278 / 0.948709 |
| Val 1972, 224 x 224 | 0.691298 / 0.546815 | 0.602600 / 0.745532 | 0.752028 / 0.854217 | 0.603358 / 0.880902 | 0.997920 / 0.829102 |

Train-mode CE fell from 0.692602 on step 1 to 0.376682 on step 20. The improvement classification uses comparable eval-mode Train CE, not the train-mode trace. Val was evaluated only before and after and never participated in gradient updates, early stopping or configuration tuning. Both crops are limited regions of single scenes and do not establish generalization accuracy. Val's last seven rows/columns and Train's last eight rows/columns remain excluded by the accepted valid crop contract.

## Resource measurements

Windows GetProcessMemoryInfo from M1-3B was reused, including process-lifetime resident working-set and private-commit peaks. Values include Python/runtime/allocator and earlier phases in each process. They are separate from tensor-capacity estimates and GPU measurements.

| Stage | Measured time |
| --- | --- |
| Model construction after initializer loading | 2.153 s |
| Optimizer construction | 0.002 s |
| Initial step forward including official CE | 0.788 s |
| Initial step backward | 1.885 s |
| Initial optimizer.step | 0.335 s |
| Save trained checkpoint | 1.130 s |
| Entire training process work | 97.738 s |
| Separate-process reload and state comparison | 6.188 s |

First optimizer-step working set rose from 1475.60 MiB to 1973.57 MiB (exact bytes in JSON). Private commit rose from 2313.29 MiB to 3084.54 MiB. During the twenty-step learning phase, sampled lifetime peaks reached 2305.18 MiB working set and 3406.25 MiB private commit. Save-phase counters are recorded separately. Fingerprinting after save can create transient memory above earlier snapshots and is not independently sampled in the training process. The reload process's lifetime peak was 2730.03 MiB working set and 4051.30 MiB private commit.

M1-3B estimated 1369.28 MiB for FP32 parameter/full-gradient/Adam moments only and did not allocate optimizer state. This experiment actually executed AdamW updates and observed whole-process memory. Those quantities have different accounting boundaries. GPU VRAM remains NOT_MEASURED.

## Trained checkpoint and restoration

External directory `C:\dev\ouem-model-cache\m1-4-cpu-20steps-01` contains m1_4_trained_step20.pth plus process JSON logs. This trained experiment artifact is distinct from the official pretrained initializer.

- Checkpoint size 1,066,468,012 bytes.
- SHA-256 0f07c0d43a02a4c33073f86899d46ed6c60bfb31f3cbaa8861079f5b77dd8047.
- Includes model.state_dict, optimizer.state_dict, completed_steps=20, configuration, seed, Torch CPU RNG state, dataset revision, input hashes/splits/license references, official initializer provenance and experiment ID.

A new Python process verified the trained-file hash before torch.load(map_location='cpu', weights_only=True), constructed the official architecture, loaded its model state strictly, restored AdamW state and Torch CPU RNG. Full tensor-content fingerprints for model, optimizer and RNG matched the first process's recorded fingerprints. Step/configuration/seed/experiment/dataset/input/provenance metadata also matched. This experiment's only random generator is Torch CPU; there is no NumPy/Python-random augmentation or CUDA state to restore.

This establishes safe load and state consistency. No optimizer step was executed in the reload process, so continuation determinism remains NOT_VERIFIED_LOAD_AND_STATE_COMPARISON_ONLY. A later bounded continuation experiment would need to compare the next step against an uninterrupted control. Exact cross-platform reproducibility remains unverified.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/m1_4_bounded_overfit.py --data C:\dev\ouem-data-cache\m1-2d-ground-01 --checkpoint C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth --output '<fresh-external-experiment-directory>'
.\.venv\Scripts\python.exe scripts/m1_4_bounded_overfit.py --reload-check --output '<same-external-experiment-directory>'
```

The fresh destination protects existing experiments. Explicit CLI execution performs heavy training; ordinary pytest uses only tiny models and synthetic labels. Existing Python 3.11.9, torch 2.7.1+cpu, torchvision 0.22.1+cpu and satlaspretrain-models 0.3.1 were reused. No dependency additions. Tests cover finite updates, AdamW checkpoint safe roundtrip, content fingerprints, metric undefined cases and the bounded configuration, with the existing loader/smoke tests retained.

The initial training runtime loaded the same training code before a small reload-only metadata-check strengthening was made. That final reload path was executed in the separate process. No extra training run or tuning occurred.

Final offline test result was 47 passed in 7.90 seconds, with 19 existing Affine pending-deprecation warnings. pip check succeeded. The new tests include a nonfinite-parameter stop check.
