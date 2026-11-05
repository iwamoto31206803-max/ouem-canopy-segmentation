# M1-2D source geometry gate and deferred resolution adaptation

## Observed outcome

The user conditionally accepted samples 661/train and 1972/val for their assigned roles. This audit retains that acceptance and the M1-2B external attribution references. It identifies a separate geometric blocker before finalizing the proposed fivefold adaptation. Status is HOLD_GSD_MISMATCH. Training-pair outputs and their hashes are absent because the requested source-geometry gate found a material difference from the target ground scale.

All recorded M1-2A original RGB, annotation, segment/COCO JSON and candidate-mask hashes were independently reread and matched. Both external raw STAC snapshot hashes matched the M1-2B manifest. STAC item IDs and CC-BY-4.0 licenses matched. Raw source attribution/contact information stays in the external cache.

## Source spatial evidence

| Property | 661 | 1972 |
| --- | --- | --- |
| Original dimensions | 2048 x 2048 | 2048 x 2048 |
| Embedded CRS | EPSG:3395 | EPSG:3395 |
| Projected pixel interval | approximately 0.1 x 0.1 m | 0.1 x 0.1 m |
| Tile-center latitude | 50.032294 | 55.816840 |
| Tile-center longitude | 31.417859 | 39.004654 |
| Local ground pixel interval | approximately 0.064362 m | approximately 0.056313 m |
| Ground interval after proposed factor 5 | approximately 0.321811 m | approximately 0.281566 m |

The original TIFFs preserve ModelPixelScale, ModelTiepoint and EPSG:3395 GeoKeys. Rasterio reconstructs a north-up affine directly from these tags. Full transforms, bounds and center-distance results are in the [machine-readable audit](M1_2D_GEOMETRY_RESULT.json). Each source footprint transformed to WGS84 is contained in its exact OAM STAC item's bounding box. Thus source geographic identity and embedded extent are consistent. Viewer/Hugging Face float32 bounds were not fetched again or used to overwrite the embedded original transform.

The geodesic calculation transforms a center pixel coordinate and its one-pixel east/north neighbors from EPSG:3395 to WGS84, then measures WGS84 ellipsoidal distances. These are local sampling intervals. Original optical resolving power remains unmeasured. The advertised 0.1m/px is consistent with projected sampling in these original TIFFs. Treating it as true ground spacing would materially overstate spacing at these latitudes.

The embedded grids and geographic footprints show no gross corruption. The discrepancy concerns the proposed interpretation of nominal GSD. The Komae reference is approximately 0.484–0.486 m ground interval. A fivefold adaptation would leave substantially finer ground sampling for both selected source scenes.

## Validity evidence

Rasterio reports nodata=None and an all-255 dataset mask for each TIFF. Each has three RGB bands, and M1-2B actually viewed both images with no obvious missing/padded area. These support the already reviewed whole-tile valid convention for these two samples. A separately measured original-survey validity mask is unavailable. Future nodata/ignore cases still require explicit verified treatment. In particular ID 0 remains background under the dataset annotation convention and is separate from spatial validity.

## Feasible nominal recipe, awaiting review

The requested operation is mathematically well-defined as a nominal 10cm-to-50cm recipe. A top-left 2040 x 2040 crop removes eight source pixels on the right and bottom, then block aggregation produces 408 x 408 output. A suitable deterministic implementation would sum uint32 RGB over exact 5 x 5 blocks and round half up using floor((sum+12)/25). A block sum has denominator 25, so half-integer ties do not occur. Decode panoptic IDs/category membership before aggregation, retain known-category crowd positives, save float32 canopy counts/25 and valid counts/25, and threshold fraction >=0.5. For 25 valid source pixels the threshold corresponds to at least 13 positive pixels. Every output uses the same crop and affine scaled by five.

That nominal recipe can be implemented after review if nominal scale is accepted explicitly. It would produce the measured 0.322m and 0.282m local intervals above. To align actual ground sampling with approximately 0.50m, the transformation needs a reviewed local-scale-aware grid/resampling decision. Approximate source-to-target factors would be 7.77 and 8.88, so the fixed factor-5 block design cannot simultaneously satisfy that interpretation. No alternative transform, threshold or data repair was applied during this audit.

## Scope of completion and next decision

Source hashing, source/STAC identity, validity metadata, CRS/affine/extent, local scale calculations and source-group separation were verified. Deterministic conversion, output-grid validation, fraction/uniform/threshold tests on converted outputs and real training-pair output hashes await an approved interpretation of the target. The M1-2D pair-preparation acceptance checklist remains incomplete. This PR is a small blocker-audit result under the user's instruction to stop finalizing the conversion if source geometry exposes a material inconsistency.

Review must choose whether to accept the explicitly nominal factor-5 recipe for initial pipeline checks, or specify a ground-scale-aware adaptation for the existing two samples. No new source data are required to make that decision.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/m1_2d_geometry_audit.py --cache C:\dev\ouem-data-cache\m1-2a-audit-02
.\.venv\Scripts\python.exe -m pytest -q
```

Existing rasterio and pyproj dependencies suffice. The audit operates offline and emits a JSON observation without writing training data or source metadata. Unit tests independently check World Mercator local scale and original-hash rejection. Source records remain referenced through the existing dataset revision d97d4da0ebbb6e249ae95ac5e19656babd972eb2 and M1-2A/B manifests.

Observed offline test result on 2026-11-05 was 23 passed in 9.12 seconds, including the existing suite. Source geometry audit completed for both cached samples and reported HOLD_GSD_MISMATCH.
