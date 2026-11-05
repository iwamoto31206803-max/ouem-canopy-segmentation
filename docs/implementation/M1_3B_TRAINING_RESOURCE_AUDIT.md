# M1-3B training readiness and resource audit

## Observed CPU run

One hash-verified real Train 661 top-left 256 x 256 patch, batch 1, four CPU threads, Python 3.11.9, torch 2.7.1+cpu, torchvision 0.22.1+cpu and satlaspretrain-models 0.3.1. Official checkpoint revision/hash and pair hashes were checked through existing M1-1/M1-3A helpers. Strict official Backbone/FPN loading, fresh Upsample/Head, official CE and finite nonzero gradients in all four components succeeded. The [result JSON](M1_3B_CPU_RESOURCE_RESULT.json) records full observations.

| Operation | Measured seconds |
| --- | --- |
| Checkpoint deserialization plus model construction | 2.391 |
| Forward including official CE | 0.703 |
| Backward | 2.164 |
| Total including imports, input/hashes and hardware query | 9.303 |

Official Head combines logits, CE and reduction inside forward, so separate CE-only timing was not instrumented. A single run measures feasibility; throughput distributions and optimizer-step cost await subsequent work.

Windows GetProcessMemoryInfo reports process working set (resident memory), process committed private memory, and process-lifetime peaks. The script samples before loading, after checkpoint loading, model-ready, forward/loss and backward. These are whole-process values including imports, inputs, allocator and transient construction memory.

| Stage | Working set MiB | Committed private MiB |
| --- | --- | --- |
| Before checkpoint load | 302.24 | 766.20 |
| Checkpoint loaded | 788.60 | 1512.66 |
| Model ready | 1134.92 | 2070.77 |
| After forward and loss | 1346.16 | 2138.85 |
| After backward | 1445.60 | 2279.56 |

Final peak working set was 1445.60 MiB and peak committed private memory 2286.87 MiB. Peaks include any earlier stage of this Python process. Shared libraries, OS paging and caches affect resident readings. Memory attributable to individual tensors cannot be inferred by subtracting these counters. CPU memory does not measure CUDA/ROCm VRAM. Optimizer state was not allocated.

## Model parameters and tensor capacity estimates

| Component | Parameters | Trainable candidate parameters |
| --- | --- | --- |
| Backbone | 87,930,848 | 87,930,848 |
| FPN | 836,608 | 836,608 |
| Upsample | 819,712 | 819,712 |
| Head | 149,890 | 149,890 |
| Total | 89,737,058 | 89,737,058 |

All parameters were unfrozen for this audit. At FP32, stored parameters require 358,948,232 bytes (342.32 MiB). The full gradient capacity estimate is the same. Actual allocated gradient tensors after this backward totalled 353,069,032 bytes. Some model parameters are unused by the official feature path, so full parameter capacity is a conservative tensor-allocation estimate.

For FP32 AdamW, estimate two FP32 moment tensors at 717,896,464 bytes (684.64 MiB). Parameters + full gradient capacity + moments total 1,435,792,928 bytes (1369.28 MiB / 1.34 GiB). This excludes activations, buffers, optimizer step scalars/temporaries, foreach/fused implementation effects, workspaces and allocator/runtime overhead. It is a tensor floor estimate, not a GPU capacity result. Step-state creation and parameter updates remain unmeasured.

Freezing only Backbone would leave 1,806,210 trainable parameters. Parameters remain resident, while estimated gradients and moments drop to 7,224,840 and 14,449,680 bytes respectively. Combined parameter/gradient/moment floor becomes 380,622,752 bytes (363.00 MiB). This alternative requires a separate run to verify memory, modes and frozen-gradient behavior; it was estimated only.

Activation demand generally grows with batch and image area, with additional window-attention padding/workspace effects. At the same patch size, accumulation over k sequential microbatches increases effective batch without multiplying simultaneous activation storage by k, but leaves parameter/gradient/moment floors. Doubling each patch dimension approximately quadruples spatial activation demand, subject to implementation details. Neither batch scaling nor alternate patch scaling was measured. No numerical claim of required VRAM beyond the tensor floor is established.

## Actual GPU/backend inventory

Windows reports AMD Radeon(TM) Graphics, driver 32.0.21030.2001, AdapterRAM 4,293,918,720 bytes. Win32_VideoController's AdapterRAM value is a hardware metadata field; it does not establish available dedicated training VRAM, especially for an integrated/shared-memory adapter.

Installed PyTorch reports 2.7.1+cpu, cuda.is_available() false, CUDA device count 0, torch.version.cuda=None and torch.version.hip=None. This environment provides CPU computation for the tested PyTorch path. ROCm/HIP and alternate GPU backends were not installed or executed. This result does not certify support or lack of support of the hardware under a different stack. GPU allocated/reserved peaks, step timing, AMP and optimizer-step capacity are NOT_MEASURED.

A first restricted Codex execution failed at the CIM hardware query before model work. A read-authorized rerun obtained the inventory and completed the resource measurements. The script's hardware query may require the same Windows read access. No drivers or PyTorch packages were changed.

## Proposed initial learning and evaluation contract

Use all-layer fine-tuning as the primary candidate, the existing 661 Train patch, batch 1, RGB float32 [1,3,256,256] in [0,1], long class-index targets [1,256,256] with background 0 and canopy 1, official Head.SEGMENT and CE. Preserve deterministic crop (0,0), common crop for all fields, hash/grid validation and all-valid pixels. Keep source oam_id and assigned roles strict.

Val uses the separate 1972 top-left 224 x 224 crop in eval mode. It excludes the final seven rows and columns and therefore covers a limited part of one source scene. M1-3A Val backward established wiring only. Future evaluation should use no_grad/inference_mode and report crop-specific CE and class metrics separately from Train. Multiple-patch/full-image evaluation must define coverage, avoid double-counting overlap, and aggregate by eligible pixels. Full 231 x 231 model output alignment remains unverified.

The installed official Head computes cross_entropy(reduction='none') and then loss.mean(). The CE default ignore_index=-100 produces zero loss at ignored positions, but Head.mean() still divides by all pixels, diluting loss with padding. Padding can also affect feature context and predictions. Current all-valid crops avoid this denominator/context issue. Do not introduce ignore/padding without a separately verified valid-loss denominator and evaluation mask. M1-3A's focused official-head test confirms the ignore behavior and remains in the offline suite.

For future GPU AMP, use supported device autocast and verify numerical behavior/gradients. CUDA float16 commonly requires GradScaler; bfloat16 selection requires actual device/backend support and testing. AMP can reduce some activation memory while FP32 model/optimizer storage often remains. CPU BF16 capability and speed were not measured; keep the baseline FP32 here. Accumulation is optional if a measured need for effective batch >1 arises. No measured need is established for the initial single-patch overfit.

References are the official [AMP documentation](https://docs.pytorch.org/docs/2.7/amp.html), [AdamW documentation](https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html) and [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html).

## M1-4 execution conditions

1. Start from a reviewed M1-3B result and preserve verified input/checkpoint hashes, assigned roles and valid crop contract.
2. Explicitly authorize a bounded optimizer-step/overfit run. Proposed initial optimizer is AdamW, lr=1e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8, FP32. These are starting proposals, not tuned or executed results.
3. Measure first-step memory/time, state creation, finite updated parameters and representative changes. Then allow a small fixed step count (e.g. 20) on Train only, inspect loss trend and log seed/config/checkpoint provenance. Val remains diagnostic and independent.
4. For CPU, ensure host RAM headroom beyond the observed 2.23 GiB private peak for moments, temporaries and optimizer execution. If GPU is selected, confirm supported driver/backend/device and compatible torch build first, then measure batch1/256 FP32 forward/backward/step allocated+reserved peaks. Budget extra capacity for moments and temporary step buffers. No particular GPU's capacity is certified here.
5. If actual capacity fails, try bounded patch/accumulation/AMP configuration changes with evidence, then consider a Backbone-frozen run. Record each change and verify which parameters update.

This audit establishes readiness for a bounded next-step experiment. Overfit success, step behavior and GPU feasibility are still pending actual M1-4 execution.

## Reproduce and test

```powershell
.\.venv\Scripts\python.exe scripts/m1_3b_resource_audit.py --data C:\dev\ouem-data-cache\m1-2d-ground-01 --checkpoint C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth --output docs/implementation/M1_3B_CPU_RESOURCE_RESULT.json
.\.venv\Scripts\python.exe -m pytest -q
```

No dependency additions. Offline tests cover parameter/trainable counts, byte estimates, backend detection without a GPU and process memory-counter consistency. Existing tests continue to verify real loader Train/Val separation, official loss semantics and M1-1/M1-2 behavior. Ordinary pytest constructs only small fixture models; the real Swin CPU audit is opt-in.

Observed test result was 43 passed in 8.27 seconds, with 19 existing Affine pending-deprecation warnings. Existing implementation files were untouched.
