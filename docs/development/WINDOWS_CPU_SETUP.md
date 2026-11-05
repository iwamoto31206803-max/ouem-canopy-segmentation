# Windows CPU development environment

This document records preparation for [M1-1](../implementation/M1_1_SATLAS_OFFICIAL_CPU_SMOKE_SPEC.md). It does not change that specification or the D0 design. Update the observations and pending checks below as M1-1 produces runtime evidence.

## Observed environment

Evidence source: the user reported the following results from PowerShell on GEEKOM-A6 / Windows. These are user-observed results, not a claim that Codex independently repeated the commands.

| Item | Observed result |
| --- | --- |
| Git | 2.56.0.windows.3 |
| Python | 3.11.9 |
| Virtual environment | `.venv` created successfully |
| pip immediately after venv creation | 24.0; current version not separately reported |
| torch | 2.7.1+cpu |
| torchvision | 0.22.1+cpu |
| satlaspretrain-models | 0.3.1 installed; import succeeded |
| pytest | 8.4.2 |
| `pip check` | `No broken requirements found` |
| `torch.version.cuda` | `None` |
| Git working tree before this preparation change | Clean |
| Local venv exclusion before this preparation change | `.venv/` in `.git/info/exclude` |

The repository `.gitignore` now also excludes `.venv/` for other checkouts. The existing local exclude entry can remain; no change to it is required.

Some Codex execution sessions could not discover Python. An attempt to run the existing venv interpreter also reported that its referenced base Python could not be found. Those session-specific failures are separate from the successful user PowerShell observations above. They do not establish that the user's environment is broken, nor that Codex has verified the installed environment. If this recurs, compare `sys.executable`, Python discovery and the venv base interpreter in the affected session before recreating anything.

## Reproduce in PowerShell

Run from this repository's root. The following is a reproduction recipe using the observed package versions, not a verbatim transcript of the user's commands. No model or checkpoint is needed for these checks.

### 1. Check Python and create a dedicated venv

```powershell
git --version
py -3.11 --version
```

Use Python 3.11.9 to reproduce the recorded interpreter version. The launcher may select a different installed 3.11 patch release; record any difference.

For a fresh checkout without `.venv`:

```powershell
py -3.11 -m venv .venv
```

Reuse an existing working `.venv`; do not overwrite it as part of routine verification. Calling its interpreter directly avoids PowerShell activation-policy changes.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -c "import sys, struct; print(sys.executable); print('bits:', struct.calcsize('P') * 8)"
```

Use a 64-bit interpreter for the Windows x64 CPU wheels. pip 24.0 was observed immediately after creation; this guide does not require an unbounded pip upgrade.

### 2. Install the CPU dependencies

The CPU wheel source is the [official PyTorch CPU index](https://download.pytorch.org/whl/cpu). The paired torch 2.7.1 / torchvision 0.22.1 versions are listed in the [official version instructions](https://pytorch.org/get-started/previous-versions/).

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.7.1+cpu" "torchvision==0.22.1+cpu" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "satlaspretrain-models==0.3.1" "pytest==8.4.2" --index-url https://pypi.org/simple
```

The [official Satlas package](https://pypi.org/project/satlaspretrain-models/0.3.1/) declares torch, torchvision, requests and matplotlib dependencies. pip resolves their transitive dependencies, including NumPy and Pillow. Their exact installed versions were not supplied in the user observations; do not present previously proposed auxiliary pins as measured results. This recipe pins the observed main packages but is not yet a complete dependency lock. Record and review the resolved environment before claiming exact reproducibility.

CUDA, torchaudio and geospatial/data-training packages are not required for this preparation step.

### 3. Verify the installed environment

These commands inspect versions and imports without constructing a model or downloading weights:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision, satlaspretrain_models; from importlib.metadata import version; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('satlaspretrain-models:', version('satlaspretrain-models')); print('pytest:', version('pytest')); print('torch.version.cuda:', torch.version.cuda)"
.\.venv\Scripts\python.exe -m pip freeze
git check-ignore -v .venv/
git status --short --branch
```

Compare with the observed table. A successful import and clean `pip check` establish basic environment readiness only. Git status depends on any current development changes; the historical clean-tree observation is not a promise that this command always returns a clean tree.

## M1-1 observed CPU execution, 2026-11-05

Codex successfully used the existing venv in this session. Python was 3.11.9, torch 2.7.1+cpu, torchvision 0.22.1+cpu, satlaspretrain-models 0.3.1 and pytest 8.4.2. `pip check` returned `No broken requirements found`. The installed official Model, SimpleHead, Upsample and state-dict prefix conversion sources were inspected directly.

Run the lightweight offline suite and explicitly opt into the full CPU smoke from the repository root.

The final lightweight suite completed with 8 passed in 8.38 seconds. It covers deterministic inputs, uniform masks, malformed outputs/loss, missing/nonfinite gradients, digest rejection before deserialization, safe CPU load options, absent-checkpoint reporting and official strict FPN loading with missing-key rejection.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/m1_1_satlas_smoke.py --init random --device cpu --size 128 --threads 4 --output docs/implementation/M1_1_RANDOM_CPU_RESULT.json
```

The random run obtained `STRUCTURAL_PASS / PRETRAINED_NOT_VERIFIED` in 13.61 seconds including imports and construction. With batch 1, RGB `[1,3,128,128]`, long targets `[1,128,128]`, each probability output was `[1,2,128,128]`. Mixed, all-background and all-canopy cases passed finite/range/probability-sum/loss checks and backward checks in head, FPN and backbone. Forward/backward and validation took approximately 1.36, 1.32 and 1.28 seconds respectively. Four CPU threads were selected to bound parallel resource use. Each case releases its graph and clears gradients before the next case. Peak memory remains unmeasured. The observed size needed no adjustment.

The script uses the official Model and loss, unfreezes parameters for gradient verification, and performs sequential cases without optimizer updates. A seeded CPU generator produces 8-bit RGB divided by 255 and preserves uniform masks. The ordinary pytest suite uses small tensors and a small official FPN fixture. Full Swin execution occurs through the explicit CLI command.

The [random result](../implementation/M1_1_RANDOM_CPU_RESULT.json) records component evidence, shapes, timings, versions and SHA-256 fingerprints of the installed official source files. The [environment snapshot](M1_1_CPU_ENVIRONMENT.txt) records observed resolved dependencies. The setup commands above pin the primary packages. To reproduce the complete observed dependency versions, use the snapshot as constraints for both installation commands with `-c docs/development/M1_1_CPU_ENVIRONMENT.txt`. Distribution artifact hashes and the exact Git commit corresponding to the installed PyPI distribution remain unverified.

For a locally supplied publisher-origin checkpoint, replace the placeholders before running.

```powershell
.\.venv\Scripts\python.exe scripts/m1_1_satlas_smoke.py --init checkpoint --device cpu --size 128 --checkpoint '<local-checkpoint-path>' --sha256 '<expected-64-digit-SHA256>' --checkpoint-source '<publisher-URL-and-immutable-revision>'
```

The script validates the digest before `torch.load(..., map_location="cpu", weights_only=True)`. The official constructor uses strict component state-dict loading for backbone and FPN. Upsample and Head are initialized afresh. Unsupported file structures or component mismatches produce a failure record. Missing files/dependencies produce `NOT_VERIFIED` and exit code 2, verification failures produce `FAIL` and exit code 1. Successful runs exit 0. OS-level termination may prevent a result file from being emitted and requires a separate failure record.

A CLI invocation with an intentionally absent checkpoint produced `NOT_VERIFIED / PRETRAINED_NOT_VERIFIED`. Its [result record](../implementation/M1_1_PRETRAINED_CPU_RESULT.json) captures that missing-file observation. Official checkpoint loading awaits a supplied artifact and independently obtained expected digest. A locally calculated hash identifies file bytes and requires publisher provenance for authenticity.

Attribution follows the formal specification. Ai2 publishes loader code under Apache-2.0 and designates checkpoint weights ODC-BY in its README. Upstream ImageNet weight lineage remains an unresolved review item. See the [official package](https://github.com/allenai/satlaspretrain_models) and [lineage investigation](../research/P0_SATLAS_AERIAL_INITIALIZATION_LINEAGE_v0.1.md).

## Remaining verification

- Input sizes beyond the observed 128 x 128 case.
- Official checkpoint acquisition, source revision, SHA-256 verification and safe loading of both backbone and FPN.
- Peak CPU memory use on GEEKOM-A6.
- Equivalence between the installed distribution and a pinned upstream Git revision.

Structural verification is complete for the recorded synthetic configuration. Pretrained verification and void/nodata/ignore-label semantics remain pending under the M1-1 specification and subsequent data QA gates.

## Updates during M1-1

Keep the user-reported preparation evidence above distinct from later agent-observed runs. Add the following when actually measured:

- Run date, execution environment, interpreter, package versions, source revision and resolved dependency record.
- Offline unit-test results and separately invoked CPU smoke commands/results.
- Input/output/target shapes, component loading and gradient checks, runtime/memory and exceptions.
- Checkpoint origin, immutable revision and expected/observed SHA-256, without committing weights.
- Separate structural and pretrained status as required by the formal M1-1 specification. Structural success alone must retain `PRETRAINED_NOT_VERIFIED` until the checkpoint smoke succeeds.

## M1-1 official checkpoint verification, 2026-11-05

This observed run supersedes the earlier absent-checkpoint observation and pending pretrained status above. The existing smoke implementation was run from main base `c5abe91` after PR #11 merged.

The official README links `Aerial_SwinB_SI` to `allenai/satlas-pretrain/aerial_swinb_si.pth`. The current main revision was resolved through the publisher API and checked again through its revision endpoint with `blobs=true`.

- Full selected revision `b1f8de04dacf9f9ed9640fe32484947710b47ede`.
- Publisher SHA-256 `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036`.
- Downloaded local SHA-256 `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036`.
- Publisher and local size both 504,439,674 bytes.
- Fixed download URL https://huggingface.co/allenai/satlas-pretrain/resolve/b1f8de04dacf9f9ed9640fe32484947710b47ede/aerial_swinb_si.pth?download=true
- External cache `C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth`.

Selection follows the current official README/loader target and freezes its resolved main revision. Older revisions can contain different bytes. Historical revisions were not exhaustively compared. The selected bytes passed the official constructor's strict loading for both backbone and FPN. The publisher file metadata distinguishes SHA-256 from the Xet storage hash. Only SHA-256 was used for content verification.

References are the [official README](https://github.com/allenai/satlaspretrain_models/blob/main/README.md), [fixed distribution page](https://huggingface.co/allenai/satlas-pretrain/blob/b1f8de04dacf9f9ed9640fe32484947710b47ede/aerial_swinb_si.pth) and [revision metadata](https://huggingface.co/api/models/allenai/satlas-pretrain/revision/b1f8de04dacf9f9ed9640fe32484947710b47ede?blobs=true).

```powershell
$revision = 'b1f8de04dacf9f9ed9640fe32484947710b47ede'
$checkpoint = "C:\dev\ouem-model-cache\satlaspretrain\$revision\aerial_swinb_si.pth"
.\.venv\Scripts\python.exe scripts/m1_1_satlas_smoke.py --init checkpoint --device cpu --size 128 --checkpoint $checkpoint --sha256 f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036 --checkpoint-source "https://huggingface.co/allenai/satlas-pretrain/resolve/$revision/aerial_swinb_si.pth" --output docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json
```

Observed `PRETRAINED_PASS`, exit 0, in 12.83 seconds including imports, hashing, safe CPU deserialization and construction. Python 3.11.9, torch 2.7.1+cpu, torchvision 0.22.1+cpu and satlaspretrain-models 0.3.1 were recorded. Backbone and FPN loaded strictly. Upsample and SEGMENT Head initialized freshly. All three cases produced `[1,2,128,128]` probabilities with finite values and valid class sums, finite official CE losses and finite nonzero Head/FPN/Backbone gradients.

| Mask | Official CE loss | Case seconds |
| --- | --- | --- |
| Mixed | 0.693265319 | 1.419 |
| All background | 0.701357603 | 1.317 |
| All canopy | 0.684756279 | 1.325 |

The updated [result JSON](../implementation/M1_1_PRETRAINED_CPU_RESULT.json) contains component evidence and source fingerprints. Lightweight offline pytest completed with 8 passed in 8.98 seconds. Peak CPU memory remains unmeasured. Other input sizes, void/ignore-label semantics and exact upstream Git revision for the installed PyPI distribution remain open as described above.
