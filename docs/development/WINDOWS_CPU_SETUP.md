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

## Not yet verified

- Official Satlas model construction and `Aerial_SwinB_SI` backbone/FPN/head wiring.
- Synthetic two-class forward, loss and backward, including gradients in backbone, FPN and head, and uniform-mask cases.
- Supported input/output sizes; 128 x 128 remains an initial smoke candidate.
- Official checkpoint acquisition, source revision, SHA-256 verification and safe loading of both backbone and FPN.
- CPU runtime and peak memory use on GEEKOM-A6.
- Exact transitive dependency versions and equivalence between the installed distribution and the reviewed upstream source revision.

No `STRUCTURAL_PASS` or `PRETRAINED_PASS` is asserted. These environment observations are not M1-1 implementation acceptance, training readiness, model-quality evidence, GPU feasibility or commercial rights signoff.

## Updates during M1-1

Keep the user-reported preparation evidence above distinct from later agent-observed runs. Add the following when actually measured:

- Run date, execution environment, interpreter, package versions, source revision and resolved dependency record.
- Offline unit-test results and separately invoked CPU smoke commands/results.
- Input/output/target shapes, component loading and gradient checks, runtime/memory and exceptions.
- Checkpoint origin, immutable revision and expected/observed SHA-256, without committing weights.
- Separate structural and pretrained status as required by the formal M1-1 specification. Structural success alone must retain `PRETRAINED_NOT_VERIFIED` until the checkpoint smoke succeeds.
