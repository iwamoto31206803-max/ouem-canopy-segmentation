# OAM-TCD technical audit — research notes 02: inference and integration

Status: source-code inspection only, **not runtime verification or a design decision**. Reviewed 2026-10-31. Upstream default branch examined without a frozen commit hash; freeze upstream commit before acceptance.

## Source trace
1. [scripts/predict.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/scripts/predict.py): command line rasterio input; reads first-axis resolution; if >1 m/pixel rejects it, and for pixel size above ~0.1 m attempts a 10 cm VRT resample. This is not appropriate unmodified for 50–60 cm experiments; it changes sampling and is likely sensitive to metric CRS assumptions.
2. [pipeline.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/pipeline.py): known-model registry, Hydra config, chooses Segformer/SMP/instance runner; delegates `predict` to `model.predict_tiled`.
3. [data/imagedataset.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/imagedataset.py), [data/tiling.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/data/tiling.py): default input tile 1024 pixels and minimum overlap 256; metric-CRS assumptions; source tile extraction and optional scaling to configured GSD. `TiledGeoImage` uses a blur + OpenCV linear resize when scaling, so its resampling behavior differs from our contemplated 5x area-averaging protocol.
4. [models/model.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/model.py): loop over tiles, skip nearly all-black/white tiles based on RGB mean, batch model calls, cache/post-process, optional warm restart. Skipping a white tile is not synonymous with valid image nodata masking; evaluate independently.
5. [models/segformer.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/models/segformer.py): HF Segformer model and processor loaded by `config.model.weights` and `revision`; resize disabled in processor; bilinear resize logits to tile dimensions then softmax. This is the **inference** path and differs from `models/segformermodule.py`, used in training.
6. [postprocess/semanticprocessor.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/postprocess/semanticprocessor.py): GeoTIFF cache quantizes probabilities `(255*p).astype(uint8)`; trims overlap margins (nominal half-overlap) and generates VRT from tile GeoTIFFs. No evidence here of probability averaging in this GeoTIFF-cache route. TODO warns validity for smaller images needs checking. Important: inspect corner and edge handling in source and empirically.
7. [result/semanticsegmentationresult.py](https://github.com/Restor-Foundation/tcd/blob/main/src/tcd_pipeline/result/semanticsegmentationresult.py): semantic mask/confidence output via result classes; default confidence threshold 0.2 for constructors; GeoTIFF result class invokes `gdal_calc.py` for binary output. `ProcessedResult._save_mask` writes GeoTIFF with CRS/transform copied from input and `nodata=0`, making background and nodata potentially ambiguous without a separate validity mask.

## Source-level risks to validate, not yet bugs confirmed in real use
- **Resolution and PSF:** source resampling blur and linear interpolation differ from controlled 5x decimation. Keep training and deployment GSD transforms explicit.
- **Seams:** nominal 50% of overlap trimmed from each tile. Edge conditions may result in seams, gaps or dimensional inconsistencies; test synthetic coordinate rasters and probability ramps, including odd overlaps and small images.
- **Confidence:** uint8 cache means <=256 distinct encoded levels rather than float32 probabilities; quantify effect on threshold calibration.
- **Nodata:** skip_empty uses per-tile mean <=1 or >=254; imagery may legitimately have bright regions. Nodata=0 in output can conflate background with absent data.
- **Threshold:** source defaults 0.2 (rather than assuming 0.5); distinguish model softmax, canopy threshold and GT-fraction threshold.
- **Reproducibility:** model inference path explicitly supports model revision; record resolved model revision SHA and image processor revision. Training `SegformerModule` uses `model_name` inherited from backbone `nvidia/mit-b0` and does not automatically use the configured `weights` repository in the inspected training function. Confirm actual initial weight states before any B/C experiment.
- **Performance:** batch, tile and model memory costs depend on runtime hardware; no measured CPU/GPU throughput yet.

## Initial boundary proposal (not adopted)
- Model repo: testable RGB-only model core, exact pixelwise preprocessing, inference contract, train/eval, outputs with probabilities and QA fixtures.
- OUEM: geographic data/CRS/GSD, spatial window scheduling and mosaicking, nodata masks, polygonization, height integration, VoxCity. Evaluate whether a minimal adapter around official model reduces dependency cost versus installing complete official pipeline.
- Preserve one golden suite comparing: single-tile inference, tiled inference, boundary consistency and geospatial alignment. This is a candidate requirement only.

## Remaining audit
- Inspect detailed `TiledGeoImage` coordinate behavior, caches/VRT and instance semantics with tests.
- Evaluate U-Net vs SegFormer in actual model cards and weight licenses; compare clean-weight pathways.
- Check HF dataset schema and 5-fold fidelity on verified records.
- Pin source commits / revisions, run independent reproduction, and defer architectural decision until then.
