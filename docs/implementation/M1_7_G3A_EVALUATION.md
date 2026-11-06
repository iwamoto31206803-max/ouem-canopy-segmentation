# M1-7 G3-A — Fixed Val3 comparison

Implementation/CPU verification only. **Real GPU metrics and real prediction/visualization quality are NOT_VERIFIED** until the user runs the reviewed commit. No training, optimizer, threshold search, new imagery or GSI inference is part of this entry point. G2 code, lineage, checkpoints, data, G0 manifest and prior receipts remain unchanged.

## Fixed inputs and models

- Dataset `ouem-zenodo10-e3db510b1125931d`, generation `zenodo-11617167-v1.0.0`.
- Manifest SHA256 `c2f4ccd56da8770cd73ef1db299d67d5528dcab8ea0051da1c2d5f8174a850ca`.
- Official initializer SHA256 `f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036`.
- Accepted G2 step20 SHA256 `47044e414c27718d37ea2d67a89f88cc24e2777757db07cfad2286ead2b9e767` (user-provided acceptance pin; Codex did not access that real GPU checkpoint).
- Both use official Aerial_SwinB_SI/Swin-B/FPN/Upsample/SEGMENT2, Satlas0.3.1, FP32, TF32/AMP off.
- **A: RECONSTRUCTED_STEP0**. Mirror G1-B/G2 threads/seed0/flags then G2 reseeding immediately before safe initializer load and the identical official Model constructor. G2's private sampler draws and memory measurements consume no model RNG and are omitted. Constructor loads Backbone/FPN strictly; all resulting tensors are additionally checked against the official prefix-adjusted state. Upsample/Head are fresh. No optimizer is constructed. **The original G2 step0 Head/Upsample snapshot is absent; exact identity is NOT_VERIFIED**, even if seed/version/order appear equal. Python/torch/torchvision/backend differences and unrecorded original-process details limit the comparison.
- **B: G2_STEP20**. Hash before `weights_only=True`, CPU map. Reuse G2's metadata validator for kind/config/architecture/seed0/dataset/initializer/completed20/schedule/history consistency; compare model fingerprint, strict-load the complete state, and compare its fingerprint again before and after GPU transfer/evaluation. Optimizer/RNG contents are not restored or advanced. Evaluation is a new commit/runtime, so training-resume environment equality is not imposed; actual differences are recorded. Satlas version and its model/head source hashes must match G2.

The official Head returns probabilities rather than logits. A removable read-only forward hook on `head.layers` observes the original pre-softmax tensor, without changing architecture/Head/loss. Argmax and class1 softmax are computed from those logits; CE is computed over all target pixels and cross-checked against official forward loss. The official probability output is cross-checked too. Unexpected spatial size, dtype, nonfinite values or disagreement fail-stop; no resizing.

## Fixed regions and metrics

Only Val IDs **1972,269,3785**, in that order, top-left `[x=0,y=0,w=224,h=224]`. No random crop, augmentation, padding or ignored background. Existing G0 verification checks all40 actual product bytes/grids/labels/validity (Train integrity reads are not Train evaluation); G1-B crop helper is called with role `val` only. Every evaluation has50,176 valid pixels, including background0. Inputs are RGB float32/255 `[1,3,224,224]`, long targets `[1,224,224]`. Uniform targets are untouched.

CPU data-only audit, **not model predictions or accuracy**:

| Val | Full H×W | Crop pixels | Excluded right/bottom pixels | Crop GT canopy fraction |
| --- | --- | --- | --- | --- |
| 1972 | 230×230 | 50,176 | 2,724 | 0.6039939523 |
| 269 | 410×410 | 50,176 | 117,924 | 0.2895806730 |
| 3785 | 328×328 | 50,176 | 57,408 | 0.1288265288 |

Regions, full shapes, excluded counts, dtype/range, product hashes and valid_fraction range are recorded in provenance. This evaluates a small fixed subset of each image, especially269; it is not whole-image evaluation.

Per scene/model: TP/FP/FN/TN, positive-class IoU, F1/Dice, precision, recall, CE, GT/predicted canopy fractions, signed error `(FP-FN)/N` and its absolute value. Prediction is `argmax(logits,dim=1)` (exact ties go to class0); no tuned cutoff. IoU=`TP/(TP+FP+FN)`, F1/Dice=`2TP/(2TP+FP+FN)`, precision=`TP/(TP+FP)`, recall=`TP/(TP+FN)`. Zero denominator → JSON null plus a reason, never0/1. Thus a correctly predicted all-background scene has undefined tree IoU/F1, not perfect tree overlap.

- **Macro**: unweighted arithmetic mean of each defined per-image rate/error, excluding null; contributing image count and all-null reason are recorded per metric. Different model precision denominators can yield different contributing counts; inspect them.
- **Micro**: sum TP/FP/FN/TN, then recompute rates/errors. Absolute global area bias can cancel opposite per-scene biases; macro mean absolute error is distinct.
- **Aggregate CE**: explicitly separate `ce_valid_pixel_mean = sum(scene_ce*N)/sum(N)`; no ambiguous macro CE field. All three actual crops have equal N. Background remains in CE/counts.
- **B−A**: per-scene and macro/micro fractional rates/errors ×100, labelled **percentage_points**, not relative-percent improvement. Counts are pixel differences; CE remains loss units. Undefined differences are null with reason.

Neither negative nor positive deltas determine pass/fail. Three fixed crops cannot establish generalization, nationwide behavior or GSI accuracy; A is reconstructed rather than a proven exact pretraining baseline. No Val-driven learning/threshold/step selection.

## Artifacts and failure rules

Fresh local output and fresh caller-supplied Drive backup are required for evaluation. Existing destinations are rejected before execution, including new child paths beneath any existing G2 `lineage.json` ancestor. Inputs are read-only. Preflight uses a separate fresh local output and does not infer metrics, construct a real Swin, or produce predictions.

- `metrics.json`: per-image, macro/micro, pixel-mean CE and B−A.
- `provenance.json`: Git/software/CUDA/GPU, pinned data/weights/checkpoint, G2 kind/step/config/training environment, A reconstruction limits, crop/policies and per-model unchanged-state fingerprints.
- `val_1972.png`, `val_269.png`, `val_3785.png`: native224 panels, deterministic names. RGB, GT, A/B masks, B FP/FN, B class1 probability, optional A FP/FN and legend. Green=Tree/TP, black=binary background, magenta=FP, orange=FN, darkgray=TN. Probability is fixed grayscale0–1, not per-image autoscale. Rendering never changes evaluation arrays.
- `result.json`: overall evaluation/copy state and payload hashes/copy checks.
- `sha256.json`: all six data/result artifact hashes and copy checks. It excludes itself to avoid self-reference; its local/backup bytes are separately SHA-compared.

Both models are `eval()` / `requires_grad_(False)` / inference_mode; no optimizer, backward, RNG restoration or G2 continuation claim is used. Complete tensor-state fingerprints before/after inference detect parameter/buffer mutation. Evaluate models sequentially to avoid keeping two models on GPU. A failure returns nonzero and local `result.json=FAIL_STOPPED`, retaining partial outputs for diagnosis. Only a report owned by this fresh run can be finalized/invalidated; no old artifact is overwritten. If a late backup failure leaves provisional remote files, a best-effort `FAILURE.json` marks them invalid. **Acceptance requires exit0, PASS result, complete matching checksum inventories in both locations and no FAILURE marker**. If storage itself is unavailable a remote failure marker may not persist: local failure takes precedence. Do not use partial directories or automatically retry into them.

## Colab cells — user runs after review/merge

Use a new free GPU runtime (T4 when available); manually mount Drive. No Codex Google authentication/upload or GPU execution occurred. Obtain the full reviewed G3-A commit; this separate checkout does not replace G2 files. All paths below are variables.

```python
from pathlib import Path
import subprocess, sys, json, hashlib, shutil, os
COMMIT = '<full-reviewed-G3-A-commit-SHA>'
assert len(COMMIT)==40 and all(c in '0123456789abcdef' for c in COMMIT)
repo = Path('/content/ouem-canopy-segmentation-g3a')
if not repo.exists():
    subprocess.run(['git','clone','https://github.com/iwamoto31206803-max/ouem-canopy-segmentation.git',str(repo)],check=True)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
subprocess.run(['git','-C',str(repo),'fetch','origin',COMMIT],check=True)
subprocess.run(['git','-C',str(repo),'checkout',COMMIT],check=True)
from google.colab import drive
drive.mount('/content/drive')
assert os.path.ismount('/content/drive')
base = Path('/content/drive/MyDrive/OUEM_Canopy_Segmentation')
dataset = Path('/content/zenodo10-g0')
receipt = json.loads((repo/'docs/implementation/M1_7_G1A_TRANSFER.json').read_text())
if not dataset.exists():
    zip_path = base/'02_datasets/zenodo10_g0/zenodo10-g0.zip'  # adjust actual upload location
    subprocess.run([sys.executable,str(repo/'scripts/m1_7_dataset.py'),'unpack','--archive',str(zip_path),
                    '--expected-zip-sha256',receipt['zip_sha256'],'--output',str(dataset),
                    '--result-json','/content/g3a-transfer-01.json'],check=True)
```

Inspect dependencies independently. Known ipython7.34/jedi diagnostics alone do not block evaluation. Review other conflicts before continuing; resolve genuinely relevant import/operator problems and record changes. **Do not unconditionally reinstall CUDA torch/torchvision**. Only if necessary, install missing Satlas0.3.1/auxiliary dependencies deliberately, then recheck; preflight enforces Satlas0.3.1 and imports.

```python
check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=False)
print(check.stdout, check.stderr, 'returncode=', check.returncode)
import torch, torchvision, satlaspretrain_models, numpy, rasterio, PIL
from importlib.metadata import version
print(sys.version, torch.__version__, torch.version.cuda)
print({n:version(n) for n in ['torch','torchvision','satlaspretrain-models','numpy','rasterio','Pillow']})
assert version('satlaspretrain-models')=='0.3.1' and torch.cuda.is_available()
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
initial = Path('/content/aerial_swinb_si.pth')
init_sha = 'f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036'
if not initial.exists():
    source = base/'01_pretrained/aerial_swinb_si.pth'
    assert sha(source)==init_sha
    with source.open('rb') as src, initial.open('xb') as dst: shutil.copyfileobj(src,dst,1024**2)
assert sha(initial)==init_sha
# Point at the accepted step20, never step10 or G1-B. No lineage file is written.
trained = base/'03_experiments/m1-7-g2-01/step20/zenodo10_g2_step20.pth'  # adjust run folder
assert sha(trained)=='47044e414c27718d37ea2d67a89f88cc24e2777757db07cfad2286ead2b9e767'
EXP = 'm1-7-g3a-01'
common = [sys.executable,str(repo/'scripts/m1_7_g3a_eval.py'),'--data',str(dataset),
          '--initializer',str(initial),'--trained',str(trained),'--experiment-id',EXP]
preflight = Path('/content/g3a-preflight-01')
subprocess.run(common+['--preflight-only','--output',str(preflight)],check=True)
assert json.loads((preflight/'result.json').read_text())['status']=='G3A_PREFLIGHT_PASS_EVALUATION_NOT_RUN'
```

Preflight validates environment, all40 products, Val crops, hashes and trained metadata/model fingerprint, but does not prove the actual GPU forward or construct/strict-load the Swin model. The following separate invocation repeats gates and performs the full evaluation. Both local/Drive destinations must be new, **outside the G2 lineage**.

```python
output = Path('/content')/EXP
backup = base/'03_experiments'/EXP
subprocess.run(common+['--output',str(output),'--backup',str(backup)],check=True)
result = json.loads((output/'result.json').read_text())
assert result['status']=='G3A_EVALUATION_AND_COPY_PASS'
assert not (backup/'FAILURE.json').exists()
inventory = json.loads((output/'sha256.json').read_text())
for name,h in inventory['sha256'].items():
    assert sha(output/name)==sha(backup/name)==h
assert sha(output/'sha256.json')==sha(backup/'sha256.json')
metrics = json.loads((output/'metrics.json').read_text())
print(metrics['model_a']['aggregate'],metrics['model_b']['aggregate'],metrics['delta_b_minus_a'])
# Inspect the actual output PNGs in Colab/Drive, then return metrics/provenance/result
# and checksum evidence for review. Do not infer visualization quality from exit0 alone.
```

If a run fails, preserve its JSON/partial outputs; diagnose before choosing another fresh experiment ID/path. No automatic crop/precision/model/threshold change. Source G2 checkpoint and lineage stay read-only. This task adds no G3-B GSI inference, retraining, dataset expansion or accuracy certification.

## Codex checks and remaining gate

**211 passed,522 warnings** (18 new cases, including parameterized boundaries), no regression in G0/G1-A/G1-B/G2. Coverage: perfect/FP-only/FN-only/background/canopy and undefined metrics, macro/micro/pixel CE/pp deltas, raw-logit CE and tie argmax, fixed Val crop/dtype/alignment, generation/hash guards inherited from G0, bad checkpoint kind/config/actual tensor shape, deterministic reconstructed seed/component checks with tiny models, inference/eval modes, state mutation rejection, fresh-path/lineage protection, artifact/provenance/hash/copy consistency and early/late failure reports.

CPU emulation uses small models and fake CUDA calls only; no real Swin or accepted GPU checkpoint is required in pytest. Actual data audit revalidated all40 existing files and loaded only Val3 for evaluation inputs. Synthetic figure layout was actually viewed; it is not a real Val prediction. **Real GPU execution, actual A/B CE/IoU/F1/area deltas, step0 identity, prediction quality and real Drive copy remain NOT_VERIFIED.** User-provided G2 acceptance is not relabelled as G3 evaluation evidence.
