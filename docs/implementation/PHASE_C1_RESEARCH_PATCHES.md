# Phase C1 direct 224px research crops

Issue #69 authorizes one exploratory crop from each of Train 1054/2493/3206 and
Val 1241/205. The baseline is main at
`9e91b0e196a2d47d4474ca886976189e4c19f27b`. Outputs use a separate research schema
and retain Formal R1 HOLD. This route calls existing low-level conversion
functions and does not invoke the R1d real-mode entrypoint.

`scripts/research_patches.py` reuses the Phase B integer floor/ceil positive-area
support calculation, `component_report`, `output_grid`, `reduce_area`, `binary`
and existing source rights checks. Every source has one bounds-preserving
approximately 0.50m grid. All stride1 224 windows avoiding edge-connected exact
black support are eligible. A per-image `random.Random(f'{seed}:{image_id}')`
selects one row-major candidate. Seed 0 was declared before the pilot.

The selection covers the full output grid. It can choose outside the largest
clear square. Interior isolated black pixels remain candidates. White exclusion
is a separate sensitivity count and chosen-window flag. Near-black/white and
source coverage uncertainties remain open. The algorithm creates no formal
validity assertion or valid_fraction product.

The source TIFF/PNG hashes and fixed roles must match the pinned Phase B JSON.
Local official Train metadata, original metadata-object identity and explicit
CC BY 4.0 STAC producer/licensor declarations are checked. Attribution and source
references are retained. This supports the bounded exploratory use, with final
commercial clearance unassessed and R1 review pending.

RGB is area averaged and rounded as in M1-2D. The teacher is the upstream
nonzero panoptic RGB union. Canopy fraction is area averaged then cast to
float32 before thresholding at 0.5. All three products share the shifted crop
affine, CRS, bounds and shape. Whole-grid intermediate arrays exist only in
memory. Each crop contains rgb.tif, canopy_fraction.tif and canopy_mask.tif.
Readback checks pixels/dtype/grid and the saved fraction threshold. Uniform
teachers are retained. Segment-level category crosswalk remains NOT_VERIFIED.

## Run and artifacts

```powershell
.\.venv\Scripts\python.exe -B scripts/research_patches.py `
  --root '<ZENODO_ROOT>' `
  --diagnosis '<PHASE_B_DIR>/patch150_feasibility.json' `
  --diagnosis-sha256 6643f2b4037cc967eeac63d2cb2460bc0c2107f568c47a311d9f12b849e07867 `
  --evidence '<STAC_DIR>' --output '<NEW_EXTERNAL_PAIRS_DIR>' --seed 0
.\.venv\Scripts\python.exe -B -m pytest -q tests/test_research_patches.py
```

STAC files are `<oam_id>-stac.json`. The CLI performs no network acquisition.
The output must be fresh and outside source, evidence and repository directories.
It contains a crop_plan.json, crops.csv, one contact sheet and 15 GeoTIFFs.
An exception retains a FAIL crop plan and any incomplete output for inspection.
Reusing that directory is rejected. Source inputs are hash-checked again after
conversion. The plan records original hashes, roles, attribution, source/full/crop
grids, seed/policy, candidate counts, white sensitivity and product hashes.

## Local pilot on 2026-11-14

All five pairs completed with matching product readback in a 6.78-second CLI
wall-time observation. The output directory is kept outside Git under
`phase-c1-pilot-20261114-01/pairs`. It contains 15 GeoTIFFs and 1,422,743 bytes
including plan/CSV/preview. No real repeat set was generated. Two synthetic sets
verify deterministic product bytes and crop selection.

| ID | split | crop xywh | binary canopy fraction | white support cells |
|---|---|---|---|---|
|1054|Train|53,49,224,224|0.000000|0|
|2493|Train|92,41,224,224|0.124980|0|
|3206|Train|122,42,224,224|0.648856|0|
|1241|Val|162,99,224,224|0.005839|0|
|205|Val|80,111,224,224|0.214684|0|

Every selected crop has zero boundary-black support cells. For 2493, y=41 is
above the prior largest square's y=70, demonstrating direct-window selection.
The contact sheet shows RGB, green positive mask with black background, and
overlay. No major artificial black cutline or large displacement was observed
in this AI visual check. Fine crown completeness, including vegetation along
the watercourse in the all-background 1054 patch, remains NOT_VERIFIED. Human
quality review is pending. The current five selected crops avoid the white
sensitivity cells too. Other sources can still require white/cutline review.

1054/1241 reuse local rights evidence. Three identified official STAC GETs
supplied missing producer/licensor evidence for 2493/3206/205, each HTTP 200,
9,182 response-body bytes total, with no redirects/retries or image/COG fetch.
Raw JSON stays in the external evidence folder. Providers are INGC, ADF Hatiti,
StEER, Lapig/UFG and Krotalo25 as declared in the individual records.

## Validation results

The final synthetic suite passed 13 tests. Full pytest passed 558 tests with
1 skipped and 4,093 warnings in 197.40 seconds. On Windows, use a fresh short
external `--basetemp` directory. The initial default Sandbox temporary path
hit WinError 206. A repository-local temporary path also triggers existing
external-output guards. Neither run was counted as a successful regression run.
The successful full run used `--basetemp=C:\dev\ouem-c1-tests-20261114`.
The temporary test sources are synthetic. No legacy implementation was changed.

The pilot's ten input hashes and fifteen product hashes were independently
checked again and matched. The Phase B diagnosis hash remains pinned above.

## Next small connection

After review, extend per-source seeded sampling to the Phase B candidate list
without materializing every overlapping window. A research data adapter can
expose RGB/target and fixed role/crop metadata to a future Trainer. The open
R2a-1 PR and the existing R1a PASS loader retain their own contracts. Source
rights, remaining quality flags and actual stored-product verification must
accompany any larger research run. Phase C2 generation and GPU training require
their own reviewed scope.
