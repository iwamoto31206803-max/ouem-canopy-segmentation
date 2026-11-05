# M1-4B deterministic CPU checkpoint continuation

## Acceptance and exact comparisons

Observed 2026-11-05. STEP20_BASELINE_MATCH, STEP21_CONTINUATION_MATCH, PROVENANCE_VERIFIED and RESOURCE_MEASURED were achieved. Both independent process records are committed as [reference result](M1_4B_REFERENCE_RESULT.json) and [resumed result](M1_4B_RESUMED_RESULT.json). Comparison policy was strict equality from the start, with no numerical tolerance or post-hoc relaxation.

The official initializer, original M1-4 trained checkpoint and all eight existing input-product hashes were reverified. The trained checkpoint remained 0f07c0d43a02a4c33073f86899d46ed6c60bfb31f3cbaa8861079f5b77dd8047 before and after both processes. Config, seed, completed_steps=20, dataset revision, input identities/splits, initial provenance and experiment ID were validated. Python 3.11.9 and torch/torchvision/Satlas versions exactly matched the M1-4 record.

Reference execution rebuilt the official pretrained model with seed 0 and four CPU threads. It repeated initial Train/Val eval/no_grad calls in the original manifest order, then twenty AdamW updates on the same Train 661 top-left 256 patch. After step 20 it repeated final Train/Val eval, matching the original save point. Evaluations use eval mode and consume no Torch RNG here. Val never participates in updates. Before step 21, reference model, optimizer and Torch CPU RNG fingerprints exactly matched the M1-4 saved-state fingerprints. The original file's matching SHA-256 ties those expected fingerprints to the preserved checkpoint.

Only after this gate passed, reference execution switched back to train mode and performed step 21. The separate resumed process loaded the original checkpoint with weights_only=True on CPU, constructed the official architecture, strictly restored model state, restored AdamW state, and restored CPU RNG after construction. It independently confirmed the same step-20 fingerprints before performing step 21 exactly once.

| Step-21 comparison | Result |
| --- | --- |
| Train-mode pre-update CE | Both 0.32750430703163147, exactly equal |
| Every named parameter gradient, including absent gradients | Full-content fingerprint identical |
| Entire updated model state | Identical |
| Entire updated AdamW state | Identical |
| Post-step Torch CPU RNG | Identical |
| Actual changed parameter names | Same 419 parameters, identical list fingerprint |

All gradients in the four components, model parameters and optimizer tensors remained finite. Actual changed parameter names were determined by hashing each full tensor before and after step 21. Zero/absent-gradient or forward-unused parameters were not required to change. The model state fingerprint covers all parameters/buffers, including unchanged tensors. Gradient comparisons cover actual content, dtype, shape and named association. State fingerprints reuse the M1-4 SHA-256 tensor-content implementation. Optimizer state includes step counters, moments and parameter groups.

## Runtime and resources

| Process | Optimizer updates | Total seconds | Step-21 measured seconds | Peak working set MiB | Peak private commit MiB |
| --- | --- | --- | --- | --- | --- |
| Reference | 1 through 21 | 101.578 | 6.202 | 2303.21 | 3405.45 |
| Resumed | 21 only | 13.391 | 6.165 | 2059.98 | 3391.63 |

Step-21 timing includes forward/CE, backward, finite checks, full-state fingerprinting and update inspection. Pre-update tensor fingerprints are prepared before that step timer. Total time includes all setup and hashing. Windows process-lifetime peaks and stage snapshots reuse M1-3B GetProcessMemoryInfo. An 8 GiB sampled private-commit ceiling is retained. Counters include runtime, allocator and measurement overhead. GPU VRAM remains NOT_MEASURED. No new model artifact is written; only JSON records are produced in fresh external directories.

## Conditions and limitations

CPU FP32, torch.set_num_threads(4), seed 0, batch1, Train256, same AdamW configuration, all layers trainable, no scheduler/augmentation/padding. The CONFIG.steps=20 field preserves the source experiment configuration. The new verification code explicitly adds only the authorized step21 in the reference series and performs only step21 in the resumed series.

Runtime flags were recorded, not changed. torch deterministic_algorithms was false and mkldnn was enabled. Strict agreement was observed under those original conditions. No deterministic-algorithm flag was enabled to manufacture a different baseline. This establishes one-step continuation identity on this specific CPU/software stack. Other CPUs, library versions, thread counts, GPU backends, repeated independent trials and longer resumed trajectories remain unverified. No future repeatability guarantee is inferred from this single paired comparison.

## Reproduce

Use fresh directories outside Git. The original trained checkpoint is opened read-only and is never a destination.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/m1_4b_continuation.py --series reference --data C:\dev\ouem-data-cache\m1-2d-ground-01 --initial C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth --trained C:\dev\ouem-model-cache\m1-4-cpu-20steps-01\m1_4_trained_step20.pth --output C:\dev\ouem-model-cache\m1-4b-reference-01
.\.venv\Scripts\python.exe scripts/m1_4b_continuation.py --series resumed --data C:\dev\ouem-data-cache\m1-2d-ground-01 --initial C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth --trained C:\dev\ouem-model-cache\m1-4-cpu-20steps-01\m1_4_trained_step20.pth --output C:\dev\ouem-model-cache\m1-4b-resumed-01 --reference C:\dev\ouem-model-cache\m1-4b-reference-01\result.json
```

The reference result must report STEP21_MEASURED and STEP20_BASELINE_MATCH before the resumed series is useful. Any strict baseline/state/loss/gradient discrepancy raises an error and yields FAIL_STOPPED. A step-20 mismatch aborts before step21. Safe hash checks precede all deserialization. No downloads or dependency changes were required.

Offline pytest completed with 52 passed in 11.19 seconds and 19 existing Affine pending-deprecation warnings. New tests use a tiny stochastic Dropout model to compare uninterrupted and restored updates, including gradients, model/AdamW/RNG fingerprints. They reject incorrect config, completed step, checkpoint hash and even a tiny nonzero loss mismatch. Existing tests remain part of the suite and real Satlas training is invoked only explicitly. pip check and diff whitespace checks succeeded.
