# D5-B1 CPU checkpoint audit

The audit covers the archived Step 280 checkpoint from `d5-v5-main280-01`. It checks serialized state from the pinned trainer at `30c3d9a63543550bfe7321f5a7dad11e646e3b94`. The original checkpoint and receipt are opened for reading. The audit writes one new private JSON report.

## Offline Windows procedure

Run this only after independent review authorizes the actual audit. Keep the already verified checkpoint and its original `runtime.json` in private local storage. Pin the receipt hash from its trusted copy and compare the local file to that pin. Obtain the actual lineage ID from independent main-run evidence such as the original run record, console binding, or reviewed private notebook output. If unavailable, stop until it can be established. Values copied from the checkpoint under inspection provide no independent binding evidence.

Prepare a private `expected.json` with these fields. Replace the lineage placeholder with the independently verified 32-character lowercase hexadecimal value.

```json
{
  "lineage_id": "<original main-run lineage ID>",
  "seconds": 900,
  "vram_bytes": 12884901888
}
```

Optional fields are `environment` (the original complete environment object), `parent` (the original parent object or verified null), and `backup_checkpoint` (the original absolute Colab path). Include them only when supported by independent records. Missing values are reported as UNVERIFIED. Prepare an independently checked SHA-256 pin of the finished contract. The CLI checks the frozen v5 binding and exact saved schedule via their canonical digest pins. It requires no v5 pixels or product files.

Use a 64-bit CPU environment with the repository's tested dependencies and Python 3.11 or later. Allow roughly 1.07 GB disk for an existing checkpoint copy and prefer a host with at least 32 GiB installed RAM. The tool measures actual available RAM and refuses to load below eight times checkpoint size (approximately 7.95 GiB). This is a preflight heuristic. The full checkpoint's measured peak is pending. Use one process at a time, keep generous headroom, and stop on resource errors. The output filesystem must have at least 1 MiB free.

```powershell
$Private = '<private local directory outside Git>'
$ReceiptPin = '<reviewed runtime.json SHA-256>'
$ContractPin = '<reviewed expected.json SHA-256>'
& .\.venv\Scripts\python.exe -B scripts/research_checkpoint_audit.py `
  --checkpoint (Join-Path $Private 'checkpoint_280.pth') `
  --size 1066503375 `
  --sha256 eeabeb671e0f0adeffd75267f06de752987f3ab2e51651875febd5a46cd6790b `
  --receipt (Join-Path $Private 'runtime.json') --receipt-sha256 $ReceiptPin `
  --contract (Join-Path $Private 'expected.json') --contract-sha256 $ContractPin `
  --output (Join-Path $Private 'step280-cpu-audit-01.json') `
  --trusted-local-checkpoint
$AuditExit = $LASTEXITCODE
```

Exit 0 and `CPU_SERIALIZED_STATE_VERIFIED` in a complete report establish the mandatory CPU checks. Exit 1 or an absent, incomplete, or FAIL report requires investigation before another attempt. Use a fresh report path for any approved follow-up. Keep all checkpoint bytes, contracts and reports private.

The fingerprint checks compare the five serialized components against the fingerprints stored in the checkpoint. Model tensors and AdamW slots are checked for structure and consistency. Slot tensor signatures cannot establish the model's exact parameter-to-slot mapping. Python and CPU RNG states are checked in isolated generators. Saved CUDA RNG byte tensors receive structural checks. Strict model/optimizer reconstruction, CUDA RNG activation, GPU restore and Val18 evaluation remain separate gates.
