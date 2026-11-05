# M1-2D ground-scale-aware training pairs

## Decision and measured outcome

The user adopted approximately 0.50m local ground spacing and conditionally accepted the existing 661/train and 1972/val samples. This supersedes the fixed-fivefold proposal evaluated in PR #16. The source audit's historical HOLD remains preserved in its own record.

Both source TIFFs remain EPSG:3395, north-up, 2048 x 2048. Their original outer bounds are preserved without crop or reprojection. At the extent center, inverse projection and WGS84 geodesic distances determine source east/north pixel intervals. Output width/height are independently rounded half up from source dimension * source ground interval / 0.50. A new affine divides the unchanged extent by those integer dimensions. Outer edges use corner coordinates; each pixel center is at (column+0.5,row+0.5).

| Sample / role | Output H x W | Ground east m | Ground north m | Error from 0.50m |
| --- | --- | --- | --- | --- |
| 661 / Train | 264 x 264 | 0.499294940 | 0.499294917 | approximately -0.000705 m |
| 1972 / Val | 231 x 231 | 0.499261303 | 0.499261275 | approximately -0.000739 m |

These are center-local intervals with small within-tile scale variation. They are within 0.15% of the target. Strict equality to 0.50m and a new projection are unnecessary for this initial near-target recipe. Optical resolving power remains unmeasured.

## Source and validity gates

All recorded raw RGB, annotation, segment/COCO JSON and candidate-mask hashes were reverified, along with external STAC snapshot hashes. Item identity, CC-BY-4.0, assigned split, group separation and containment in the STAC footprint are rechecked by the existing geometry audit. The two STAC records contain attribution information retained externally. Row and annotation licenses and source references are carried into the [result manifest](M1_2D_PAIR_RESULT.json).

Both GDAL dataset masks are all-valid, with nodata=None, and M1-2B visually reviewed the full tiles. The approved two-sample whole-tile-valid assumption is reused. valid_fraction is generated independently from an all-one validity raster and saved separately. ID 0 denotes dataset background, not invalidity. Future inputs with nodata or uncertain validity fail the gate and need a separately reviewed policy.

## Deterministic transformation

A separable pixel-overlap matrix computes exact box-area weights for arbitrary rational source/output dimension ratios. This applies a box low-pass before subsampling and handles partial source pixels at output edges. All aggregation uses float64. RGB is rounded with floor(value+0.5), clipped to [0,255], and stored as RGB uint8. It shares exactly the same output shape and source-edge footprint as every teacher field.

Panoptic IDs are decoded and validated through existing M1-2A logic. Only mapped Canopy 1 and Tree 2 become positive, including the accepted known-category crowd pixels. Unknown IDs/categories are rejected. The source binary mask is area-averaged, clipped to [0,1] and stored as float32 canopy_fraction. Threshold >=0.5 is applied to that saved-representation fraction and stored as uint8 0/1. valid_fraction uses the same weights and float32 format. Uniform-mask behavior is tested. Fraction is projected-grid area coverage; the local projection-scale variation over these small tiles is retained as a limitation rather than introducing a complex geodesic-area integration.

Each sample folder contains four GeoTIFFs sharing CRS, affine, bounds, width and height. RGB has three bands and the remaining fields one band each. Every file was reopened and compared for exact array equality, exact affine/CRS/shape and bounds within 1e-8 projected metres. Only after both samples complete does the script write the success manifest. Failed runs can leave partial external files without a success manifest; use a fresh destination on retry.

## Actual execution and reproducibility

```powershell
.\.venv\Scripts\python.exe scripts/m1_2d_prepare_pairs.py --cache C:\dev\ouem-data-cache\m1-2a-audit-02 --output C:\dev\ouem-data-cache\m1-2d-ground-01
.\.venv\Scripts\python.exe scripts/m1_2d_prepare_pairs.py --cache C:\dev\ouem-data-cache\m1-2a-audit-02 --output C:\dev\ouem-data-cache\m1-2d-ground-02
.\.venv\Scripts\python.exe -m pytest -q
```

Both real runs completed and all eight output file SHA-256 values matched. The result JSON records those hashes and full source/output grids. Same-environment determinism is verified. Cross-platform BLAS/library variation and GeoTIFF encoder-version variation remain untested. No dependency additions were needed. Recorded NumPy, Pillow, rasterio and pyproj versions are included in the result.

Synthetic tests cover noninteger area weights against a hand-derived result, average-area conservation, uniform fractions, threshold boundaries, deterministic channel-preserving reduction, source extent preservation, local GSD and bad assigned split rejection. Existing tests cover unknown IDs, source-group overlap and input hash failures. Real conversion validates all output readbacks and grid consistency. These tests do not establish model accuracy or full-dataset readiness.

## Handoff and scope of acceptance

Two real pairs meet the initial preprocessing PASS-candidate conditions under the user's accepted rights and validity assumptions. 661 remains Train and 1972 remains Val. The old manifest's review flag is historical; this result explicitly records the user's acceptance for these assigned roles. Exact original survey validity masks, optical resolution and annotation completeness remain unavailable. Wider data still requires record-level QA and rights checks. Variable output dimensions are allowed here; later training input crop/padding and batching decisions belong to the training pipeline.

Output files remain in the external caches. Git contains only scripts, synthetic tests and metadata/results. This change supplies preparation artifacts for Satlas and keeps model implementation separate.

Final test result was 30 passed in 9.60 seconds, with nine Affine pending-deprecation warnings from affine multiplication (including rasterio internals). pip check succeeded. A third run using the final script also matched all eight output SHA-256 values.
