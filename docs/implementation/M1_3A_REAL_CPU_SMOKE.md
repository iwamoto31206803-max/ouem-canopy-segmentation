# M1-3A real OAM-TCD to official Satlas CPU smoke

## Observed acceptance

2026-11-05, Python 3.11.9, CPU torch 2.7.1+cpu, torchvision 0.22.1+cpu and satlaspretrain-models 0.3.1. The existing M1-2D ground-01 products were used directly. All eight file hashes matched the PR #17 result manifest. Each CRS, affine, bounds, dimensions, band dtype and value domain was checked. Masks matched thresholded fractions and validity was all one. Assigned roles and source-group separation were preserved.

[Actual run JSON](M1_3A_REAL_CPU_RESULT.json) records REAL_DATA_CPU_PASS. All samples used batch 1, four CPU threads, seed 0 and deterministic top-left crop shared across RGB/mask/fraction/validity.

| Sample | Role / model mode | Patch | Official CE | Forward/backward/check seconds |
| --- | --- | --- | --- | --- |
| 661 | Train / train mode | 256 x 256 | 0.692602456 | 3.208 |
| 1972 | Val / eval mode with gradients | 224 x 224 | 0.691298008 | 2.704 |

Total time including imports, file/hash checks and construction was 15.277 seconds. Both cases returned [1,2,H,W] probabilities with finite values in [0,1], near-one class sums and finite official CE. Finite nonzero gradients appeared in Backbone, FPN and Head. The Val backward is an isolated wiring check after gradients were cleared. No optimizer exists and no parameter update occurs.

## Initialization and provenance

The official Aerial_SwinB_SI file was locally SHA-256 verified against the M1-1 checkpoint record before `torch.load(map_location='cpu', weights_only=True)`. The official Model's strict Backbone/FPN loaders succeeded. Upsample and two-class Head initialize freshly. Revision b1f8de04dacf9f9ed9640fe32484947710b47ede and SHA-256 f8abf21cadaab2a71859e9d172bec4e5d6b7cc9710882a2f6a4d8aae826db036 are included in the result.

The result also fingerprints the authoritative pair manifest and installed official Head source. Dataset revision and rights/attribution lineage remain in the referenced M1-2D result and M1-2B external STAC records. No data or checkpoint bytes are committed. Original venv was reused without dependency additions.

## Input contract and safe Val treatment

The loader requires RGB uint8 [3,H,W], mask uint8 [1,H,W] with values 0/1, and finite float32 fraction/validity [1,H,W] in [0,1]. It rejects nodata, validity below one, inconsistent threshold masks and grid disagreements. RGB becomes torch.float32 [1,3,H,W] divided by 255 in original channel order. Teacher becomes torch.long [1,H,W]. Every field uses crop origin (0,0).

Official installed code inspection shows Swin patch/hierarchical stages followed by FPN and factor-four Upsample from highest-resolution features. Arbitrary input dimensions must not be assumed to produce same-sized labels. Sizes 256 and 224 were actually verified in this run. The entire 231 x 231 Val image was not executed. The chosen 224 crop is divisible by 32 and fits completely within its valid extent, excluding the last seven rows/columns. Train excludes the last eight rows/columns of its 264 image. Georeferencing of the patch uses the unchanged top-left source output affine and the smaller patch extent. Full-image Val evaluation is deferred.

Official SimpleHead('segment') calls cross_entropy(reduction='none') then averages over all pixels. The PyTorch default ignore_index=-100 makes ignored pixel losses zero, but the final mean retains the full image denominator. A focused test confirms this installed official path. The initial smoke uses only valid crops, with no padding/ignore labels. Future padded batching needs an explicitly reviewed valid-loss reduction before use. Fractional valid pixels are rejected by this loader.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/m1_3a_real_cpu_smoke.py --data C:\dev\ouem-data-cache\m1-2d-ground-01 --checkpoint C:\dev\ouem-model-cache\satlaspretrain\b1f8de04dacf9f9ed9640fe32484947710b47ede\aerial_swinb_si.pth --output docs/implementation/M1_3A_REAL_CPU_RESULT.json
```

The heavy model is invoked only through the explicit smoke CLI. Offline tests use tiny local GeoTIFF fixtures and a small official Head. They reject bad hashes, mismatched grids, label values, dtypes, fractions, validity, bands/shapes, wrong split and oversized/padded patch requests. They check deterministic crops, RGB/GT coordinate agreement and official ignore-loss behavior. Existing M1-1 and M1-2 tests remain part of the suite.

## GPU handoff and limitations

The CPU real-data wiring prerequisite is satisfied for the two recorded patches. GPU micro-step can reuse this contract, starting batch 1 and measuring allocated/reserved memory and step time before choosing training patch/batch/freeze configuration. GPU feasibility is unverified. CPU process memory was not instrumented in this run and remains NOT_MEASURED. Losses are diagnostics from fresh task heads. Model accuracy, full-image validation and invalid-region semantics remain outside the evidence obtained here. The CLI records failures and returns nonzero when validation or official execution raises an exception.

Final offline suite completed with 40 passed in 10.53 seconds. Nineteen Affine pending-deprecation warnings were recorded from existing geometry code, rasterio internals and fixture construction.
