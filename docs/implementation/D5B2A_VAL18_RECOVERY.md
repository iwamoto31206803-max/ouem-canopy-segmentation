# D5-B2a fixed Val18 recovery

This procedure requires a separate owner decision after code review. First authorize one Step 280 / Val 1241 CPU pilot. Review its memory use and latency before authorizing any full assessment.

The prior GEEKOM-A6 verification covered the Step 280 checkpoint and receipt. Confirm local offline availability of the frozen raster products, crop plan and companion JSONs, and the official initializer before attempting the pilot.

## Prepare offline on Windows

- Use a clean checkout that contains Git object `30c3d9a63543550bfe7321f5a7dad11e646e3b94`, Python 3.11 or later, CPU-compatible PyTorch/torchvision, Rasterio, Pillow and SatlasPretrain 0.3.1. The tool verifies historical Python source contents against the pinned Git tree, with LF/CRLF normalization. It does not install packages or download inputs.
- Keep the frozen v5 directory, three companion JSONs, approved initializer, checkpoint, and its original receipt in private input directories. Place new output and any local copy in separate sibling trees outside the repository and input directories. Use a quiet host, and choose a RAM margin, thread count, timeout and free-disk minimum before execution. Example bounds below use 12 GiB currently available RAM, 128 MiB free output space, four CPU threads and 600 seconds. These are proposed pilot bounds. CPU peak and latency remain unmeasured. The RAM and disk checks are preflight checks, and the parent watchdog limits the evaluation child wall time. Windows records OS peak working set and peak pagefile commit charge as a private-memory indicator. Evaluation timing excluding rendering includes data reads, evaluator checks and aggregation.
- Prepare a private reviewed pin contract `step280-pins.json`. Replace the evidence reference with a non-sensitive identifier for the independent records. Pin the contract SHA from the reviewed bytes. Include optional `lineage_id` only if independently evidenced by the actual executed run. The preflight UUID is unsuitable.

```json
{
  "step": 280,
  "size": 1066503375,
  "sha256": "eeabeb671e0f0adeffd75267f06de752987f3ab2e51651875febd5a46cd6790b",
  "receipt_sha256": "57dbefa89730adb270f17120079ecc3799f4da73b834be838a015d724d981835",
  "evidence_ref": "Reviewed Step280 archive and original receipt"
}
```

```powershell
$Inputs = '<private input directory>'
$Reports = '<existing separate private reports directory>'
$PinSha = '<reviewed lowercase pin-contract SHA-256>'
& .\.venv\Scripts\python.exe -B scripts/research_val18_recovery.py `
  --execute --pilot --recovery-id d5b2-step280-pilot-01 `
  --dataset (Join-Path $Inputs 'dataset-v5') `
  --review (Join-Path $Inputs 'research_review.json') `
  --freeze (Join-Path $Inputs 'dataset_freeze_v5.json') `
  --candidates (Join-Path $Inputs 'candidates.json') `
  --initializer (Join-Path $Inputs 'aerial_swinb_si.pth') `
  --checkpoint (Join-Path $Inputs 'checkpoint_280.pth') `
  --receipt (Join-Path $Inputs 'runtime.json') `
  --pins (Join-Path $Inputs 'step280-pins.json') --pins-sha256 $PinSha `
  --output (Join-Path $Reports 'd5b2-step280-pilot-01') `
  --threads 4 --timeout-seconds 600 --min-ram-bytes 12884901888 --min-disk-bytes 134217728
$RecoveryExit = $LASTEXITCODE
```

Pilot output is `PILOT_ONLY` and `non_acceptance: true`. Review the provenance, CPU resource readings and valid pixel count (50,176) separately from any full result. For each separately approved assessment use one command with `--assess --step 0|70|140|210|280` (choose one step), a fresh recovery ID and new output path. For Step 0 omit all four checkpoint/receipt/pin options. For 70/140/210 obtain independently reviewed archive sizes, SHA-256 values and original receipt SHA-256 values before use. Those checks remain pending. Recovered checkpoints are `RECOMPUTED_FROM_CHECKPOINT`. Step 0 is `RECONSTRUCTED_STEP0`, with verified official backbone/FPN mapping and unverified equality of the original random head.

## Preserve a completed step

Require exit 0 and a complete `completion.json`. A full assessment contains 18 canonical 1120×224 PNGs, `metrics.json`, provenance, and a relative-name SHA index. The receipt pins the index SHA. The tool flushes and publishes the source receipt before publishing the copy receipt. If the copy receipt fails, the verified source receipt can remain, the copy is incomplete and the command fails. A partial receipt never qualifies as complete. Use fresh approved paths for a follow-up. Destinations must support exclusive hardlink publication (for example local NTFS).

Optional `--local-copy <fresh separate path>` verifies copied bytes and records `LOCAL_COPY_READBACK_PASS`. A synced or mounted path still reports `CLOUD_DURABILITY_UNVERIFIED`. Before the next full step, have the owner independently inspect the non-shared Drive destination and its access settings, list all objects, and download each object through authenticated Drive access into a fresh directory independent of the sync cache. Compare exact inventory and per-file SHA to the local index, and compare index and completion-receipt bytes to the local originals. Record the remote folder identity, verification UTC, observed byte hashes and sharing check in a separate private remote receipt. Use `REMOTE_DRIVE_DURABILITY_CONFIRMED` only after these checks. Publish no private output or links to GitHub.

Recovery outputs describe new inference. CPU/CUDA numerical equality, original Step 0 equality and historical per-update losses remain unverified. GPU restore and Formal R1 acceptance are separate.
