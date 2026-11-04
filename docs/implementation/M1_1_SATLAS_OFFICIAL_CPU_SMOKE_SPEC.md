# M1-1 — SatlasPretrain official CPU smoke: implementation specification

**Status:** Proposed implementation handoff, 2026-11-04. Documentation-only. No code, package installation, checkpoint download, or runtime PASS is asserted.
**Authority:** [Project Charter](../PROJECT_CHARTER.md), [Design Principles](../DESIGN_PRINCIPLES.md), [Satlas-first ADR](../decisions/ADR_20261101_SATLAS_FIRST.md).
**Next work:** Codex implements M1-1 in a **separate, small implementation PR**, reviewed before merging.

## 1. Purpose and acceptance boundary

Demonstrate that the official Ai2 SatlasPretrain aerial single-image Swin-v2 Base architecture can be instantiated on **Windows / Python 3.11 / CPU**, and that an official two-class semantic segmentation forward, CE loss and backward step work with **synthetic RGB and 0/1 canopy masks**.

Two different evidence levels must be kept separate. **Implementation PR disposition:** if only the structural smoke can run because the official checkpoint is not yet available, report `STRUCTURAL_PASS / PRETRAINED_NOT_VERIFIED` and treat the M1-1 implementation as **partially verified**, not fully accepted for progression to real-data training:
- **Structural smoke (random weights, no download):** proves compatible model/head/loss wiring only, **not Satlas checkpoint loading**.
- **Pretrained smoke (locally supplied official checkpoint):** additionally proves that the intended `Aerial_SwinB_SI` backbone **and FPN** can be loaded and used with a newly initialized two-class segmentation head. Never label the first as proof of the second. **Pretrained smoke must pass before moving to any training work that assumes Satlas initialization**, even if a structural-only implementation PR is reviewed and merged.

**Not M1-1:** OAM-TCD data access/labels, any GSI inputs, ground resolution choice, GPU benchmark, real-data training, model-quality metrics, comparison models, checkpoint redistribution, commercial license signoff or OUEM geospatial integration.

## 2. Primary-source implementation assumptions to check

- Official package: `allenai/satlaspretrain_models` Python package `satlaspretrain_models`. Pin the exact dependency and source revision once a compatible Windows/Python-3.11 setup is found; do not blindly rely on unbounded upgrades.
- Model identifier: `Aerial_SwinB_SI`. Official `Model(num_channels=3, multi_image=False, backbone=Backbone.SWINB, fpn=True, head=Head.SEGMENT, num_categories=2, weights=...)` is the **candidate API**, to be verified in the pinned version. Backbone and FPN weights are pretrained only if a real official checkpoint is supplied and loaded successfully; segmentation head remains newly initialized.
- RGB: `float32`, `[B,3,H,W]`, 8-bit RGB normalized by `/255` to `[0,1]` per Ai2 high-resolution imagery guidance. Do not add ImageNet mean/std unless the official aerial path explicitly requires it.
- Class IDs: **0 = background**, **1 = Tree/Canopy**; synthetic targets `torch.long` with shape `[B,H,W]`. `Head.SEGMENT` (not `Head.BINSEGMENT`) uses two-channel logits internally and returns softmax probabilities. Its `forward(images, targets)` returns `(probabilities, loss)`.
- The official FPN + upsample + `SimpleHead('segment')` implementation is **not** a general interchangeable U-Net. Use the official path first; no custom decoder or loss in this step.
- The official segmentation loss currently averages cross-entropy over target pixels. **Void/nodata/ignore index is NOT approved or validated** by M1-1: test clean 0/1 synthetic masks only; log the risk for M1-2 instead of silently remapping void to background.
- The model's shape constraints (e.g. divisibility), torchvision compatibility and resulting spatial dimensions must be established by an actual smoke, not assumed.
- Official `Weights.get_pretrained_model(...)` downloads weight bytes and calls `torch.load`; for a controlled offline/secure smoke, prefer constructing the **official Model** from a user-supplied local state dict, loaded safely (see below). No wrapper around the public API should silently change FPN/head semantics.

## 3. Implementation request (bounded)

1. Inspect current repository and upstream sources first. Implement a small, independently runnable CPU-only smoke entry point (suggestion: `scripts/m1_1_satlas_smoke.py`) plus focused pytest tests. Avoid reorganizing historical inference/PoC code.
2. Expose two explicit initialization modes (CLI names may differ if documented):
   - `--init random`: **offline** official architecture with `weights=None`; never report pretrained PASS.
   - `--init checkpoint --checkpoint <local_path> --sha256 <expected_digest>`: load only the explicitly supplied official checkpoint; no automatic HTTP fetch, and reject absent/mismatching digest. Record download origin/immutable revision where known. Computing a digest locally alone records file identity, **not independent authenticity**.
3. Prefer `torch.load(path, map_location="cpu", weights_only=True)` for locally supplied, publisher-origin tensors, with an explicit safe failure if this is incompatible with the pinned version. Do not use `weights_only=False` or unpickle arbitrary files just to make the test pass. Do not commit weights/data or dynamic download artifacts.
4. Exercise deterministic synthetic RGB input and 0/1 masks; assert finite probabilities/loss, `[B,2,H,W]` output where supported, near-one class probability sums, and finite backward gradients for the newly initialized segmentation head **and at least one trainable parameter in each of the backbone and FPN**. These checks prove gradient propagation, **not optimizer updates or full fine-tuning readiness**. If a component is frozen, explicitly unfreeze it for this isolated gradient check or clearly mark the check unverified. Document any exact input-size constraints. Cover **all-background and all-canopy** masks without changing pixels.
5. Verify the intended **backbone AND FPN** really load in checkpoint mode (strict state-dict compatibility or explicit missing/unexpected-key checks; never silently accept random pretrained components). Fail clearly if the downloaded artifact's structure differs from expectations. A structural smoke is not evidence for weight loading.
6. Add a short Windows `py -3.11` / venv / CPU instructions section, with tested install dependencies/version details. Keep the normal `pytest` checks **offline and lightweight**; put full Swin-v2 Base forward/backward in an explicit opt-in smoke, not every unit-test run. **Use 128×128 as an initial synthetic input-size candidate, not an assured compatible size**; if the official model rejects it or CPU resources constrain execution, change to a minimal supported size, document why, and avoid silently weakening the tested contract. Missing dependencies, checkpoint or CPU resource exhaustion must be logged as **NOT VERIFIED**, not reported as PASS.
7. Produce a concise machine-readable or Markdown result record including mode, package/source revisions, Python/torch/torchvision versions, input/output/target shapes, checkpoint source+SHA if applicable, model component loading, loss/backward status and any exceptions. Separate planned tests from actual observed runs.

Minimal expected user commands (adjust to the implemented CLI, document exact commands):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\m1_1_satlas_smoke.py --init random --device cpu --size 128
.\.venv\Scripts\python.exe scripts\m1_1_satlas_smoke.py --init checkpoint --device cpu --size 128 --checkpoint C:\path\to\aerial_swinb_si.pth --sha256 <verified_digest>
```

These are interface examples, **not claims of tested commands**. Never copy checkpoint contents into a PR.

## 4. Acceptance checklist

- [ ] No changes to historical PoC outputs or existing inference defaults; no OAM-TCD/GT54 access or GSI processing.
- [ ] Package/source versions and Windows/Python-3.11 compatibility documented from actual inspection; setup is reproducible with pinned, compatible dependencies.
- [ ] Offline unit tests run with meaningful asserts and no implicit network calls.
- [ ] Structural CPU synthetic smoke executed and independently labeled `STRUCTURAL_PASS` **only if actually observed**. If the official checkpoint is absent, explicitly report `STRUCTURAL_PASS / PRETRAINED_NOT_VERIFIED` (partial implementation verification); do not label M1-1 fully verified.
- [ ] Official-checkpoint CPU smoke executed and labeled `PRETRAINED_PASS` **only if backbone+FPN weights load and forward/loss/backward succeed**; otherwise `PRETRAINED_NOT_VERIFIED` or `FAIL` with a specific reason. Do not begin Satlas-initialized data training until `PRETRAINED_PASS` is observed.
- [ ] Two-class `Head.SEGMENT` semantics and clean 0/1 masks, including uniform-mask cases, verified; backward gradients observed in the head, FPN, and backbone for the isolated unfrozen test.
- [ ] No false claim of M1 fine-tuning, real canopy performance, GPU capacity, safe commercial deployment or license G0 completion.
- [ ] Attribution/provenance notes identify code license, published ODC-BY designation and unresolved upstream ImageNet weight lineage; no final rights signoff implied.
- [ ] Changes submitted for review as **separate implementation PR**; do not auto-merge.

## 5. Handoff after M1-1

A structural-only implementation may be reviewed and merged with the explicit `PRETRAINED_NOT_VERIFIED` limitation, but this **does not complete the checkpoint-loading prerequisite**. Finish and record `PRETRAINED_PASS` before any Satlas-initialized training. M1-2 (CC BY-only OAM-TCD data/label QA and target GSD decision based on existing GSI GeoTIFF) can be prepared independently; then proceed to GPU micro-step and training configuration after the relevant prerequisites pass. CPU success is **not** evidence of GPU feasibility or segmentation accuracy.

## Official technical references

- [Ai2 SatlasPretrain package](https://github.com/allenai/satlaspretrain_models)
- [Model loader and FPN flag](https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/model.py)
- [Segment head and CE target path](https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/heads.py)
- [FPN and upsampler](https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/models/fpn.py)
- [Aerial RGB normalization](https://github.com/allenai/satlas/blob/main/Normalization.md)
- [Prior Satlas / ImageNet lineage audit](../research/P0_SATLAS_AERIAL_INITIALIZATION_LINEAGE_v0.1.md)
