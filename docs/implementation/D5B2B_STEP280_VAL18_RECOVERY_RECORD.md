# D5-B2 Step280 fixed-Val18 recovery: execution, evidence, and archival record

**Record date:** 2026-11-21 (JST)  
**Status:** `D5_B2_STEP280_FIXED_VAL18_LOCAL_EVIDENCE_COMPLETE` + `REMOTE_DRIVE_DURABILITY_CONFIRMED`; model-performance analysis pending; Formal R1 **HOLD**  
**Owner-operated runtime:** GEEKOM-A6, Windows, CPU only.  
**Scope:** The Step280 assessment of original run `d5-v5-main280-01` **only**. This record does not close out the other historical steps.

## 1. Purpose, boundaries, and source trace

The original D5 v5 280-update GPU run produced historical fixed-Val18 evaluation output in an ephemeral environment. Its metrics and PNGs were **not durably retained**. This work reconstructs **new CPU evaluation output from the archived original Step280 checkpoint**; it does **not** restore or claim byte/numeric identity with the unavailable historical GPU Val18 output.

- Pinned historical trainer and evaluation semantics: commit `30c3d9a63543550bfe7321f5a7dad11e646e3b94`; `scripts/research_cuda.py`, `scripts/research_data.py`, `scripts/m1_7_g3a_eval.py`.
- Review/authorization: [D5-B2 design and execution gates, Issue #95](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95); [D5-B2a implementation, Issue #96](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/96).
- Reviewed recovery implementation: [PR #97](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/97), squash-merged to `main` as `81d0cabaaee89e2d816689577376668d6e19a57d`.
- Implementation: `scripts/research_val18_recovery.py` and runbook `docs/implementation/D5B2A_VAL18_RECOVERY.md`; no modification to historical trainer, labels, fixed sample selection, or checkpoints.
- Codex-reported implementation validation (not independently re-executed by reviewer): 85 targeted PASS / 1 SKIP; 697 full PASS / 2 SKIP.

The recovery CLI checks historical source content using LF-normalized SHA-256, fixed dataset and schedule, externally reviewed checkpoint/receipt hashes and strict CPU model state, then invokes the pinned evaluator, confusion/aggregate calculation, and five-strip PNG renderer. Each execution uses fresh private directories and a receipt-last SHA evidence protocol.

## 2. Frozen identities and execution environment

| Evidence / contract | Verified identity |
| --- | --- |
| Frozen dataset identity digest | `a645e831c4214cab913506e01fe9160f5272cf8aed1a6b00bbbf35964cd3b6d7` |
| Fixed schedule SHA-256 | `5c3a1483e1087645e2ae472d993aae520102106da6d55a66a67ca1469661c601` |
| Crop plan SHA-256 | `b6711c33eafed51dd8d42040abf13e3a52601d5be54b08479217ac7e043c8692` |
| Research review SHA-256 | `ccda53e9dac2c40fda6be718b9fc8a57dc35e6210fcc271a5a5533e390456dbc` |
| Frozen v5 manifest SHA-256 | `d7a72bdf1c792678cee0030c72dded18d3cdd6639a800c12ebafc63ba3fa1978` |
| Candidate manifest SHA-256 | `bf71faf1ab512a326804f808aeff541f8e1363c8dece6d2250c6ebe7cbece0db` |
| Official initializer SHA-256 | `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036` |
| Step280 checkpoint | 1,066,503,375 bytes; SHA-256 `eeabeb671e0f0adeffd75267f06de752987f3ab2e51651875febd5a46cd6790b` |
| Original Step280 runtime receipt SHA-256 | `57dbefa89730adb270f17120079ecc3799f4da73b834be838a015d724d981835` |
| Locally created independent pin-contract SHA-256 | `a3dbf8b017523029557e76f232d8bf0ed9b3ff3e3607d49c2517560a6873e423` |

Owner's GEEKOM-A6 preflight independently reported all five source-input file hashes PASS, the historical Python code check PASS, frozen dataset identity/schedule PASS, and Val1241 product shape and binary-teacher integrity PASS. The frozen schedule contained **280 training updates and 18 fixed evaluation windows**.

Actual CPU evaluation environment recorded in private `provenance.json`: Python **3.11.9**, PyTorch **2.7.1+cpu**, torchvision **0.22.1+cpu**, SatlasPretrain **0.3.1**, CPU/FP32, **4 threads**. This differs from the original GPU/Colab runtime; do not claim numerical identity with it.

## 3. Ordered execution record

1. **Offline preparation (PASS).** Confirmed Git main includes PR #97, fixed v5 data and official initializer exist locally (a duplicate of the frozen v5 ZIP and initializer also exists in private Drive), and checkpoint/receipt match reviewed original hashes.
2. **One-window Step280 / Val1241 CPU pilot (PASS).** Owner separately authorized and ran `--execute --pilot`; exit code 0, `PILOT_ONLY`, `non_acceptance=true`. The local `1241.png` five-strip and metrics were produced without GPU/training. Reported preflight available RAM 15.79 GiB; peak working set 2,298,662,912 bytes and peak private commit 3,928,784,896 bytes. This pilot is **not** one of the full recovered assessments.
3. **Pilot visual check.** The owner supplied the five-strip Val1241 image for private inspection. A descriptive diagnostic was recorded in Issue #95, but its apparent land-cover failure causes and generalization remain hypotheses; no labels or thresholds were changed.
4. **Separately authorized Step280 fixed Val18 CPU assessment (PASS).** Executed `--execute --assess --step 280` **once**, with the same pinned inputs, 4 threads, 600-second child watchdog, minimum 12 GiB available RAM, and minimum 128 MiB output disk. Local run yielded `FULL_VAL18_VALIDATION: PASS`, exact ordered **18 rows**, **903,168** evaluated pixels (18 × 224 × 224), **18** five-strip PNGs, `scope=FIXED_VAL18_RECOMPUTATION`, `classification=RECOMPUTED_FROM_CHECKPOINT`, and `non_acceptance=false` (meaning full assessment rather than pilot; **not Formal R1 acceptance**).
5. **Local evidence verification (PASS).** New private local output plus separate private readback copy. `LOCAL_COPY_READBACK_PASS`; `completion.json` and `sha256.json` present and verified by the CLI and owner checks.
6. **Remote Drive preservation (PASS).** Owner manually uploaded the 22 evidence files to the existing private OUEM Drive hierarchy. Reviewer independently enumerated and downloaded every Drive object through authenticated Drive access, not from GEEKOM-A6's sync cache; verified object inventory and hashes; then matched the original GEEKOM-A6 anchor SHA-256 hashes to remote downloads. See Section 5.

Execution dates, source/head IDs, timings and device details are stored in the private `provenance.json`. These are owner-operated execution claims corroborated by verified evidence artifacts, not a separately rerun reviewer-side model inference.

## 4. Recomputed Step280 Val18: immutable recorded measurements, no performance interpretation yet

- Exact canonical Val18 order: `1241,205,461,1575,1137,1609,4328,22,5014,5003,1950,618,1207,2006,3279,802,2861,3245`.
- Valid pixels: **903,168**; aggregated confusion: **TP 125,355; FP 28,802; FN 49,454; TN 699,557**; their sum is 903,168.
- Recorded **micro tree IoU 0.6156592718468059**, **macro tree IoU 0.40682443455533224**, **micro F1 0.7621152337931579**, **CE valid-pixel mean 0.24432695015437073**. These numbers are **archival facts**, not an acceptance criterion or a generalization finding.
- Background-only windows and undefined rates: `461` and `3279` contain no positive truth/predictions, so tree IoU/F1 etc. are null; `802` has no positive truth and ten FP, so recall is null under the original metric policy. These are documented undefined cases, not computation failures.
- Val1241's row reproduces the separately performed pilot: TP 210, FP 729, FN 83, TN 49,154; tree IoU 0.2054794520547945; CE 0.19014249742031097.
- Owner-reported full assessment timings: load 1.422 s, model construction 1.609 s, evaluation excluding render 4.984 s, render 0.375 s. Sum of these **recorded stages ~8.39 s**, not complete process wall time. Peak working set **2,298,626,048 bytes** (~2.14 GiB); peak private commit **3,928,453,120 bytes** (~3.66 GiB).

The full per-image rows, aggregates, caveats and original image data are preserved **only in private evidence storage**; do not copy the private image set into this repository.

## 5. Archival identity and independent remote readback

**Local private evidence paths (GEEKOM-A6):**
- Assessment: `%LOCALAPPDATA%/OUEM/D5B2_PILOT/d5b2-step280-val18-full-01/`
- Local readback copy: `%LOCALAPPDATA%/OUEM/D5B2_PILOT/d5b2-step280-val18-localcopy-01/`
- Reviewed pin contract: `%LOCALAPPDATA%/OUEM/D5B2_PILOT/reviewed-pins/step280-pins.json`

**Private Google Drive:** `OUEM_Canopy_Segmentation/04_results/d5b2-step280-val18-full-01`; [folder](https://drive.google.com/drive/folders/1BTf3Oap9OrwtWVwT_DgeHMnTTP3F7QWV).

Inventory/readback:
- **22/22** files: 18 canonical `<image_id>.png` files and 4 JSON files (`metrics.json`, `provenance.json`, `sha256.json`, `completion.json`). No unexpected or duplicate objects were reported.
- Authenticated remote readback: **20/20 SHA-256-indexed evidence objects match** the remote `sha256.json`; all **18/18 PNGs** decode as RGB and are **1120 × 224** (five strips).
- Drive metadata at verification reported `shared=false` for the folder and each object, owner-only permission at parent folder; this is a point-in-time sharing check.
- The independent remote `sha256.json` digest and the owner-reported local original digest **both equal**  
  `22b8ad02609fe0af78f2904daccd9660359ab390410d82d552fa533192bffce3`.
- The independent remote `completion.json` digest and the owner-reported local original digest **both equal**  
  `185dd05615c71530c71796fae07e58d4accbdf8cd1121ca50196d0ddc4bcc782`.
- The index hash agrees with the `index_sha256` claim in `completion.json`; thus the local anchor → remote completion → remote index → 20 evidence objects chain is complete.

**Attestation:** `REMOTE_DRIVE_DURABILITY_CONFIRMED` is supported by the separately recorded independent verification, **not** by modifying original output. The original `completion.json` retains `cloud=CLOUD_DURABILITY_UNVERIFIED`, correctly reflecting the status **at creation time**.

Primary trace records: [local full Val18 assessment](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95#issuecomment-7057038381), [remote file readback](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95#issuecomment-7057083071), [final local-anchor comparison](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95#issuecomment-7057122072).

## 6. Explicit limits, holds and next-phase handoff

**CONFIRMED within scope:** reviewed immutable recovery tool, pinned inputs and data identity, CPU Step280 one-window pilot, complete fixed Val18 recomputation, local evidence readback and independently confirmed private Drive remote archive.

**Not verified / not claimed:**
- Independently verified executed GPU **lineage ID** (`independent_lineage=UNVERIFIED`), or GPU `RESTORE_PASS`.
- Byte-for-byte or numerical equivalence of CPU-generated outputs to the original **unarchived** CUDA Val18 outputs; historical per-update losses.
- Step 0 original random-head equality, independently reviewed original archive/receipt pins for Steps **70/140/210**, or recovered assessments for **0/70/140/210**.
- Representative accuracy for Japanese urban tree canopy, domain-transfer success, benchmark acceptance, or Formal R1 completion. **Formal R1 remains HOLD.**

**Next planned work (not started by this record):**
1. Analyze full fixed Val18 micro/macro metrics, class prevalence, image-level variation and undefined-rate handling; inspect selected private five-strip panels to characterize TP/FP/FN and possible source-label ambiguity. Maintain original threshold and frozen identities for this diagnostic.
2. Decide whether to commission any additional step recovery. Each additional step requires separate owner authorization, independent checkpoint/receipt pins, and private remote durability verification before continuing.
3. Do not infer improvement, change training labels/thresholds, start GPU work, alter the R1 gate or announce a broader tree-crown performance result from the archival confirmation alone.

This document is a **factual phase handoff**; the interpretive model-performance review follows separately.
