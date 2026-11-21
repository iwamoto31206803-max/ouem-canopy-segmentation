# OUEM Canopy Segmentation

Engineering repository for a practical **RGB tree-canopy semantic-segmentation component** of the **Open Urban Environment Model (OUEM)**.

**Goal:** Use Japanese GSI aerial RGB imagery at approximately 0.5–0.6 m/ground pixel to produce usable 2D tree-canopy masks for OUEM's simplified 3D canopy, shade/heat and landscape workflows. The priority is a maintainable, reproducible implementation with documented rights and limitations—not a state-of-the-art benchmark or individual-tree delineation.

## Current state — 2026-11-21 D5 research learning milestone and D6 planning

**A major exploratory milestone has been reached:** the frozen v5 SatlasPretrain SwinB + FPN + 2-class SEGMENT run `d5-v5-main280-01` completed **280 GPU updates**, and its Step280 checkpoint was subsequently hash-pinned and recovered on GEEKOM-A6 **CPU**. Some original-source validation images show spatially meaningful tree-canopy extraction rather than the all-background historical Step20 behavior. This establishes **initial learning feasibility**, not practical model accuracy or a numerically controlled improvement over the previous experiment.

Research v5 remains **Train 70 originals / 139 windows; Val 18 originals / 18 fixed windows** (88 originals / 157 windows / 471 TIFFs). Step280 fixed Val18 was recomputed on CPU across **18 windows / 903,168 pixels**: micro tree IoU **0.615659**, macro tree IoU **0.406824** (16 defined windows). The full evaluation (18 PNG + metrics/provenance/SHA/completion = **22 private evidence files**) passed local-copy validation and **independent Google Drive remote SHA readback**, anchored against local original hashes. Evidence: [D5-B2 formal execution/archival record](docs/implementation/D5B2B_STEP280_VAL18_RECOVERY_RECORD.md), [PR #98](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/pull/98), [Issue #95](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/95). The original GPU Val18 outputs were not recovered, and CPU/GPU numerical equality and executed lineage independence are unverified.

**Research decision:** pause detailed Val18 failure-cause/threshold analysis; retain the current SwinB/FPN/SEGMENT2 baseline and plan for *larger, more diverse and quality-reviewed training data* plus a *bounded restartable GPU workflow*. The two **D6 planning workstreams** are **D6-A deterministic, cross-PC preprocessing/QA** and **D6-B free-tier-aware GPU/temporary-storage/checkpoint/recovery operations**. **Their order has not been chosen, and a new dataset or GPU campaign is not authorized.** See [D6 design baseline](docs/D6_PREPROCESSING_GPU_OPERATIONS_PLAN.md) and [Next Roadmap](docs/ROADMAP_NEXT.md).

**Formal R1 remains HOLD (0/150 accepted);** Japanese positive-tree validation, real-GSI accuracy, commercial sign-off and OUEM production suitability are **NOT VERIFIED**. The research CLEAR/frozen status is not an R1 acceptance. [Current State](docs/CURRENT_STATE.md) is the operational status reference; dated stage snapshots below are historical.

## Historical checkpoint — 2026-11-18 Phase C closure

**Phase C1–C8 research-data expansion and the scoped scientific review are complete.** Research candidate **v5 contains Train 70 original images / 139 windows and Val 18 original images / 18 fixed windows**, totaling 88 originals, 157 windows and 471 GeoTIFFs. Research CLEAR records Codex AI screening. **Formal R1 remains HOLD for all 150 candidates, with 0 accepted.** Source rights, bytes and private image-review findings remain Codex-local reports awaiting the targeted human handoff. [Phase C8](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/80), [scientific review disposition](https://github.com/iwamoto31206803-max/ouem-canopy-segmentation/issues/81#issuecomment-6971582740)

Historical SatlasPretrain T4 learning, Step10 continuation, Step20 restoration and Val3 inference have succeeded. The Step20 model predicted all background, with Tree IoU/F1 zero. **Useful canopy quality remains NOT MET. New GPU training with v5 is unperformed**, and v5 CUDA integration still needs a small reviewed change. GSI suitability is NOT VERIFIED and commercial use remains unapproved.

**Next work is GEEKOM image confirmation and experiment preparation.** Reconcile local v5 evidence and hashes, inspect the pinned 33-window contact sheets with targeted human judgments, then specify the fixed-Val18 exploratory experiment. Reuse the existing Satlas GPU primitives in a narrow research CUDA bridge before separately approved execution on an available, measured CUDA runtime. The existing `ResearchData` input and source-balanced sampler are CPU-validated, while `r2_train.train` enforces CPU FP32.

Read [Current State](docs/CURRENT_STATE.md) for current facts and pins, [2026-11-18 Project Review](docs/PROJECT_REVIEW_20261118.md) for evidence and limits, and [Next Roadmap](docs/ROADMAP_NEXT.md) for decision gates. The [Project Charter](docs/PROJECT_CHARTER.md), [Design Principles](docs/DESIGN_PRINCIPLES.md), and [Satlas-first ADR](docs/decisions/ADR_20261101_SATLAS_FIRST.md) retain design precedence. Earlier snapshots and GPU recipes below preserve their dated scope. The [C4 dataset / Val7 plan](docs/implementation/PHASE_C4_RESEARCH_DATASET.md) is an archived v1 snapshot.

## Historical route and review snapshot — 2026-11-07

**Active engineering route:** Ai2 SatlasPretrain `Aerial_SwinB_SI` official backbone/FPN + newly initialized 2-class SEGMENT head; fine-tune using verified CC BY 4.0 OAM-TCD RGB/Tree–Canopy labels at approximately **0.50 m/local ground pixel**, then visually assess actual Japanese GSI aerial imagery.

**Milestone status:** The bounded GPU training, checkpoint/restore and Val3 inference/evaluation **pipeline has been demonstrated** on Tesla T4. **M1 remains incomplete and useful model quality is NOT established**: the G2 Step20 model predicted zero canopy on all three fixed Val 224×224 crops (micro Tree IoU/F1 0). This is not GSI performance evidence or a final model judgment.

The 2026-11-07 review keeps the Satlas-first direction, preserves prior G0–G3 artifacts as an immutable small-run reference, and proposes **~100 total QA-accepted images (provisionally Train 80 / Val 20, with existing Zenodo10 eligible after new R1 QA)** for the first expansion. Exact folds/source separation and evidence-based QA take precedence over a numeric target. Build a generalized dataset/Trainer independently of frozen smoke tooling; perform early qualitative GSI review once a trained candidate yields nontrivial source-domain masks.

The [2026-11-07 Project Review](docs/PROJECT_REVIEW_20261107.md) preserves this earlier proposal and its evidence. Documentation PRs #41 and #65 were subsequently merged. Their checked-in status gives no run-specific execution, licensing or acceptance approval. Use the latest [Current State](docs/CURRENT_STATE.md) and [Roadmap](docs/ROADMAP_NEXT.md) for the present handoff.

| Stage | Actual status as of this snapshot | Evidence |
| --- | --- | --- |
| M1-1 through M1-4B | Windows CPU model/preprocessing and bounded continuation demonstrated | [M1-4B](docs/implementation/M1_4B_CONTINUATION.md) |
| M1-5A through M1-5D | Source inventory, Zenodo lineage and data QA pilots; unresolved HOLD sources excluded | [M1-5A / pilot records](docs/implementation/M1_5A_DATASET_INVENTORY.md) |
| M1-7 G0 / G1-A | Zenodo10 fixed Train7/Val3, products and Drive transfer hash verified | [G0 manifest](docs/implementation/M1_7_G0_MANIFEST.json) |
| M1-7 G1-B | One T4 GPU step, checkpoint/restore demonstrated | [G1-B code](scripts/m1_7_g1b_gpu_smoke.py) |
| M1-7 G2 | Train7 bounded 20-step learning, Step10 continuation, Step20 stored | [G2 code](scripts/m1_7_g2_train.py) |
| M1-7 G3-A | T4 Val3 inference, metrics, PNG visualization and Drive copy PASS; canopy-quality NOT MET | [G3-A evaluation specification](docs/implementation/M1_7_G3A_EVALUATION.md) |
| Next | Approx. 100-image data QA, general Trainer, meaningful source-domain learning, GSI qualitative suitability | [Roadmap](docs/ROADMAP_NEXT.md) |

**Reading caution:** Detailed historical operation recipes later in this README and older stage documents retain the status and environment assumptions present **when those recipes were written** (including some `GPU pending` language). For current status use [Current State](docs/CURRENT_STATE.md), then the actual run receipts/provenance. Do not treat this README update as a replacement for source SHA verification or formal rights clearance.

## Where to find the authoritative information

- **Requirements and responsibilities:** [Project Charter](docs/PROJECT_CHARTER.md), [Design Principles](docs/DESIGN_PRINCIPLES.md), [Satlas-first ADR](docs/decisions/ADR_20261101_SATLAS_FIRST.md), [OUEM handoff](docs/OUEM_HANDOFF.md).
- **Executable code:** `scripts/` (one bounded operation per stage); corresponding offline/unit checks in `tests/`.
- **Implementation results:** `docs/implementation/` (stage-specific methods, inputs, hashes, observed results and limits).
- **Environment:** [Windows CPU setup](docs/development/WINDOWS_CPU_SETUP.md) and its [observed package snapshot](docs/development/M1_1_CPU_ENVIRONMENT.txt). The setup document records **M1-1-era observations**; its historical “pending” entries should be read alongside the later M1-2 through M1-5A results above.
- **Historical research and license analysis:** `docs/research/`. The [earlier experiment plan](docs/EXPERIMENT_PLAN.md) is **superseded** where it conflicts with the active charter/ADR.

The charter and design documents retain their original decision-stage wording as part of their provenance. This README reports **current execution status**; stage reports are the authority for what was actually tested. Do not infer a PASS in a newer environment from a historical result.

## Existing code and reproducibility

The core work is **already scripted**, but there is **not yet one general end-to-end download/prepare/train command**. Earlier scripts deliberately target bounded inputs; avoid treating them as unrestricted pipelines for all 4,608 images.

The M1-6 recipe below preserves its legacy 661/1972 input scope. Linked G1-B/G2/G3-A stage procedures preserve their Zenodo10 Train7/Val3 lineage. The historical v5 GPU bridge was implemented and executed in D5; its original-source run, restored Step280 assessment and archival evidence are recorded in the D5-B2 record. D6 scaling is separately proposed, not yet implemented.

| Operation | Existing implementation | Notes |
| --- | --- | --- |
| Research patch supply | `scripts/research_patches.py`, `scripts/research_data.py` | Frozen v5 source/label processing; exact output hashes and research-only reviews; D6-A cross-PC equivalence not yet tested |
| Research v5 bounded GPU training / CPU reconstruction | `scripts/research_cuda.py`, `scripts/research_val18_recovery.py` | Original D5 280-update T4 training; Step280 CPU fixed-Val18 reconstruction and private remote evidence readback PASS; see D5-B2 record |
| Source-balanced research schedule / CPU training core | `scripts/r2_train.py` | `make_plan` supports research supply, `train` enforces CPU FP32 |
| Bounded original-data acquisition and label QA | `scripts/m1_2a_data_qa.py`, `scripts/m1_2b_overlay_qa.py` | Initial two records; enforce rights and label checks |
| Geospatial geometry and ~0.50 m data-pair generation | `scripts/m1_2d_geometry_audit.py`, `scripts/m1_2d_prepare_pairs.py` | Current pair preparation is limited to approved IDs 661/1972 |
| Verified training-patch loading | `scripts/m1_3a_loader.py` | Uses exact relative paths, hashes and grid/validity checks |
| Real-data CPU model checks and resource audit | `scripts/m1_3a_real_cpu_smoke.py`, `scripts/m1_3b_resource_audit.py` | Read stage records for invocation and limits |
| Bounded optimizer updates and continuation | `scripts/m1_4_bounded_overfit.py`, `scripts/m1_4b_continuation.py` | CPU experiments, not general training CLI |
| Bounded Colab GPU smoke | `scripts/m1_6_colab_gpu_smoke.py` | Opt-in CUDA FP32; one step, resume to 20 total, separate restore check |
| Dataset-wide metadata inventory | `scripts/m1_5a_inventory.py` | Metadata only; does not authorize or acquire all new image pixels |

**For a fresh Windows checkout:**

1. Follow [Windows CPU setup](docs/development/WINDOWS_CPU_SETUP.md) and reproduce or record the resolved Python/packages. The observed baseline was Python 3.11.9, PyTorch 2.7.1+cpu and `satlaspretrain-models` 0.3.1; other machines/configurations must be checked rather than assumed identical.
2. From the repository root, run the non-GPU regression tests:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q
   ```

3. Obtain the **source data, processed data, official initializer and trained checkpoints separately**. They are not committed to Git. Use an external data/model cache, not the repository or its virtual environment. Preserve the manifest and relative paths such as `train/661/rgb.tif` and `val/1972/canopy_mask.tif`.
4. Use the exact CLI and acceptance conditions in the linked **stage document** before rerunning an expensive, destructive or networked operation. Many stages require a **fresh output directory** and fixed source revision/hashes. Do not overwrite accepted artifacts.
5. Compare expected and observed hashes; where software-dependent GeoTIFF byte-for-byte equality is not guaranteed, also check pixel content, CRS, affine transform, extent and label semantics. Keep source and attribution records with the data.

The two existing processed pairs are documented in [M1-2D result](docs/implementation/M1_2D_PAIR_RESULT.json): `train/661` is 264 × 264 and `val/1972` is 231 × 231 before patch cropping. The fixed OAM-TCD revision is `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`. Existing files are external to the repository; a new PC must receive or regenerate them and verify the hashes.

**Historical M1-5B planning note (not the current next task):** At that stage, extend the existing acquisition/QA/preprocessing logic to small additional source groups with configurable paths, network limits, stable manifests and separate offline processing. Prefer reusing existing scripts/tests and updating existing documents over creating a parallel pipeline or numerous new files. Windows/CPU portability and non-interactive operation matter for eventual company-PC use; Google Colab GPU execution is a distinct environment-specific gate.

## Historical M1-6 procedure: bounded Colab FP32 GPU smoke (this specific CLI was not GPU-verified)

One CLI, [`scripts/m1_6_colab_gpu_smoke.py`](scripts/m1_6_colab_gpu_smoke.py), connects the existing verified loader and official Satlas model to CUDA. **Real-model T4 execution, Python 3.13/PyTorch 2.11 compatibility, FP32 VRAM fit, GPU updates and GPU checkpoint restoration are NOT_VERIFIED by Codex.** On 2026-11-05 the user separately observed Colab Python 3.13.16, torch 2.11.0+cu130, Tesla T4 (14.56 GiB), CUDA tensor operations, Drive I/O and transferred-input hashes. Those checks do not establish Satlas CUDA compatibility or training success. Free-tier GPU availability and capacity are not guaranteed.

The experiment uses only Train 661 (top-left 256 square) and Val 1972 (top-left 224 square), batch 1, RGB FP32/0–1, long labels 0/1, seed 0, four CPU threads, official SEGMENT/CE, all-layer AdamW (lr `1e-4`, weight decay `0.01`, betas `(0.9,0.999)`, eps `1e-8`). No AMP, scheduler, augmentation, padding or optimizer updates on Val. The maximum is **20 total completed optimizer steps**, including restored steps. CPU/GPU bitwise equality is not required.

### Colab cells: prepare a reviewed, pinned checkout

Select a GPU runtime. After this PR is reviewed, insert its **full reviewed commit SHA** below; do not use a moving branch as the experiment identity. Each code block is a Python cell. Use fresh local/Drive output paths for a new experiment.

```python
import os, re, subprocess, sys
from pathlib import Path
COMMIT = "<FULL_REVIEWED_COMMIT_SHA>"
assert re.fullmatch(r"[0-9a-f]{40}", COMMIT), "Supply the reviewed 40-character commit"
REPO = Path('/content/ouem-canopy-segmentation')
assert not REPO.exists(), 'Use a fresh checkout; do not reset an existing one'
subprocess.run(['git', 'clone', 'https://github.com/iwamoto31206803-max/ouem-canopy-segmentation.git', str(REPO)], check=True)
subprocess.run(['git', 'checkout', '--detach', COMMIT], cwd=REPO, check=True)
os.chdir(REPO)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
```

Check the preinstalled torch/torchvision pair **before** installing auxiliary packages. Stop on import/operator errors or missing CUDA; do not blindly reinstall PyTorch. Upstream Satlas 0.3.1 declares Python >=3.9, torch >=2.1 and torchvision >=0.16 without upper bounds; that declaration alone does not verify Python 3.13/torch 2.11. The CLI records resolved versions and official model/head source hashes. See [official package metadata](https://github.com/allenai/satlaspretrain_models/blob/main/setup.py) and [PyTorch CUDA APIs](https://docs.pytorch.org/docs/2.11/cuda.html).

```python
import torch, torchvision, importlib.util, importlib.metadata, json
print(sys.version, torch.__version__, torchvision.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'Select a GPU runtime; inspect the installed torch build'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
# Explicit auxiliary setup only: retain Colab's torch/torchvision pair.
# Record the original/resolved environment and pip install reports outside the checkout.
ENV = Path('/content/ouem-m1-6-environment')
ENV.mkdir(exist_ok=False)
(ENV/'before.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
# --no-deps prevents Satlas from replacing torch/torchvision.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                'satlaspretrain-models==0.3.1', '--report', str(ENV/'satlas-install.json')], check=True)
# pyproj is imported by the shared geometry/loader helper. Use binary wheels for this Python.
missing = [p for p in ['rasterio', 'pyproj', 'requests', 'matplotlib', 'pytest']
           if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', *missing,
                    '--report', str(ENV/'auxiliary-install.json')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
import satlaspretrain_models, rasterio, pyproj
(ENV/'after.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_m1_6_colab_gpu_smoke.py'], check=True)
```

Installation reasons are limited to missing loader/test dependencies and the accepted Satlas 0.3.1 API. If an installed auxiliary package imports unsuccessfully or `pip check` fails, record the error and investigate before running. Do not upgrade/reinstall torch to resolve unrelated Colab package warnings. No package set on Python 3.13 is claimed verified here; keep `before.txt`, `after.txt` and install reports with the run.

### Mount Drive, stage exactly the accepted inputs, verify

Drive paths below are examples, separate from CLI logic. The eight GeoTIFFs and transferred manifest are compared against the repository's accepted M1-2D record; all file hashes, CRS, affine, bounds, shape, labels and all-valid fractions must pass. Original imagery/STAC hashes and rights references are **inherited provenance**, not reverified raw files in Colab: raw originals are not part of this transfer. No new data/checkpoint download occurs.

```python
from google.colab import drive
import shutil

drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/OUEM_Canopy_Segmentation')
LOCAL = Path('/content/ouem-m1-6-inputs')
LOCAL.mkdir(exist_ok=False)
DATA = LOCAL/'data'
for split, image_id in [('train', 661), ('val', 1972)]:
    destination = DATA/split/str(image_id)
    destination.mkdir(parents=True)
    for name in ['rgb.tif', 'canopy_fraction.tif', 'canopy_mask.tif', 'valid_fraction.tif']:
        shutil.copy2(DRIVE/'02_datasets/m1-2d-ground-01'/split/str(image_id)/name, destination/name)
shutil.copy2(DRIVE/'02_datasets/m1-2d-ground-01/manifest.json', DATA/'manifest.json')
INITIAL = LOCAL/'aerial_swinb_si.pth'
shutil.copy2(DRIVE/'01_pretrained/aerial_swinb_si.pth', INITIAL)
sys.path.insert(0, str(REPO/'scripts'))
from m1_1_satlas_smoke import checkpoint_digest
from m1_6_colab_gpu_smoke import validate_inputs
checkpoint_digest(INITIAL, 'f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036')
patches, provenance = validate_inputs(DATA, DATA/'manifest.json')
print([(split, list(x.shape), list(y.shape)) for split, (x,y,_) in patches.items()])
del patches
```

Initializer revision: `b1f8de04dacf9f9ed9640fe32484947710b47ede`; dataset revision: `d97d4da0ebbb6e249ae95ac5e19656babd972eb2`. Both inputs remain read-only. Training reads staged local data once, rather than repeatedly reading GeoTIFFs over Drive.

### One optimizer step, then explicitly continue to twenty

All outputs must be new directories **outside the repository and input trees**. The first call stops after one step and saves its checkpoint. Inspect `result.json` before choosing to continue. Every step validates probability shape/range/class sum, finite CE/gradients/parameters/optimizer tensors, and an actually updated active parameter in each of Backbone/FPN/Upsample/Head.

```python
CLI = [sys.executable, 'scripts/m1_6_colab_gpu_smoke.py', '--data', str(DATA), '--checkpoint', str(INITIAL)]
ONE = Path('/content/ouem-m1-6-step1')
subprocess.run(CLI + ['--output', str(ONE), '--max-steps', '1'], check=True)
one = json.loads((ONE/'result.json').read_text())
assert one['optimizer_step_status'] == 'GPU_OPTIMIZER_STEP_PASS'
print(one['status'], one['steps'], one['memory_final'])
```

Back up this first result before continuing. Use a unique experiment label. Checkpoints include optimizer state and can be several GB; check local/Drive free space. Do not publish them or credentials to GitHub.

```python
RUNS = DRIVE/'03_runs/m1-6-run-01'  # change for each experiment; never overwrite
RUNS.mkdir(parents=True, exist_ok=False)
shutil.copytree(ONE, RUNS/'step1')
shutil.copytree(ENV, RUNS/'environment')
```

Run the next cell only after the one-step result succeeds. Restoring checks the supplied SHA-256 **before** `torch.load(map_location='cpu', weights_only=True)`, strictly loads model state, restores AdamW and CPU/CUDA RNG, and compares full-state fingerprints. No `weights_only=False` fallback is provided. `--max-steps 20` adds 19 steps to a step-1 checkpoint.

```python
TWENTY = Path('/content/ouem-m1-6-step20')
subprocess.run(CLI + ['--output', str(TWENTY), '--max-steps', '20',
    '--resume', str(ONE/'gpu_smoke.pth'), '--resume-sha256', one['saved_checkpoint']['sha256']], check=True)
twenty = json.loads((TWENTY/'result.json').read_text())
print(twenty['initial_eval'], twenty['final_eval'])
shutil.copytree(TWENTY, RUNS/'step20')
```

Verify restoration in another Python process without taking step 21:

```python
RESTORE = Path('/content/ouem-m1-6-restore')
subprocess.run(CLI + ['--output', str(RESTORE), '--restore-only',
    '--resume', str(TWENTY/'gpu_smoke.pth'), '--resume-sha256', twenty['saved_checkpoint']['sha256']], check=True)
restored = json.loads((RESTORE/'result.json').read_text())
assert restored['status'] == 'GPU_CHECKPOINT_RESTORE_PASS' and restored['completed_steps'] == 20
shutil.copytree(RESTORE, RUNS/'restore')
```

After disconnection, remount Drive, recreate the same pinned checkout/dependencies and stage the original inputs. Copy the saved `gpu_smoke.pth` and its `result.json` from Drive to a fresh local directory. Pass that JSON's `saved_checkpoint.sha256` with `--resume`. For an incomplete run use `--max-steps 20`; for a completed 20-step run use `--restore-only`. Never overwrite the original saved run. A hard session termination can lose progress since the last successful checkpoint; this CLI saves at the end of each successful invocation, not after every step. Default one-step invocation bounds that initial risk.

Environment/commit differences block restoration by default. `--allow-environment-change` is an explicit separate compatibility experiment: differences are recorded, but data/configuration/provenance, step limits and full restored-state fingerprints must still pass. Restoration does not establish GPU continuation determinism; CPU M1-4B bitwise results do not transfer to CUDA.

### Acceptance and failure diagnosis

Accept the GPU smoke only after user inspection of actual Colab records confirms: official strict Backbone/FPN initialization, new Upsample/Head; real Train/Val valid crops; finite forward/official CE/backward; active parameter changes and AdamW state; at most 20 completed steps; synchronized forward/backward/optimizer timings; allocated/reserved current/peak VRAM and device total; initial/final **eval-mode** Train/Val CE and metrics; saved checkpoint SHA-256 and a separate-process `GPU_CHECKPOINT_RESTORE_PASS`. Training output alone reports `GPU_SMOKE_STEPS_PASS_RESTORE_PENDING`. Undefined metric denominators are JSON `null`. Report Train improvement/unchanged/worsened without adjusting this bounded experiment from Val results; these two crops establish no generalization claim.

Exceptions (including OOM/CUDA errors) stop immediately and write `result.json` with failing stage, exception/traceback, available memory counters, conditions and completed steps. Exit 1 means failure; exit 2 means `CUDA_NOT_VERIFIED`; neither is a GPU PASS. CUDA timings synchronize before/after operations; peak counters cover the invocation, including validation/restore/checkpoint work, not just training activations. PyTorch allocator counters exclude some CUDA/runtime and other-process memory, and are not Windows working-set measurements. A process kill may prevent JSON finalization.

If a subprocess cell fails, inspect the new output's `result.json` and copy that directory to a **fresh** Drive failure folder even though `check=True` stopped the cell. Interrupt the runtime if it becomes unresponsive. Preserve the FP32 failure stage/VRAM/exception/configuration; do not silently switch to AMP, freeze layers or shrink patches. Consider those only as separately reviewed experiments after diagnosis. Import/API failures on Python 3.13/torch 2.11 likewise require a recorded compatibility decision before changing versions.

The accepted CPU scripts and earlier evidence are unchanged. CPU unit tests cover CLI bounds, unsafe paths, missing CUDA, failure JSON, transferred metadata rejection, safe lightweight checkpoint roundtrip/hash rejection and resume compatibility; they do not execute a real CUDA model. Final T4 acceptance is the user's post-review Colab run.

## License, evaluation and deployment cautions

Ai2 lists SatlasPretrain weights under **ODC-BY** and the model-loading library under **Apache-2.0**. The upstream ImageNet initialization chain and eventual internal/commercial deployment obligations remain open for review; these facts are **not a final legal clearance**. Use only OAM-TCD training records with verified **CC BY 4.0** dataset/annotation and original imagery-source rights and attribution. Do **not** use Restor's released noncommercial/restricted pretrained model weights in this training chain.

Keep official OAM-TCD holdout and independent Japanese references out of training and development tuning. GSI qualitative inspection does not establish quantitative target-domain accuracy; GT54's existing land-cover classes are not automatically canopy ground truth. No public trained-model/checkpoint redistribution is planned.

## Project history

This repository was formerly `oam-tcd-poc`. Prior scale/edge-effect experiments, older A/B/C adaptation plans and superseded model routes remain in history and research records, not in the current implementation mandate.
